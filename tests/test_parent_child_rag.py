"""Offline parent/child tests with real loaders and explicit golden evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from api.schemas import ChatRequest, IntentResult, KnowledgeHit, KnowledgeSection, RagQualityCase
from agents.customer_service_agent import CustomerServiceAgent
from rag import knowledge_base
from rag.hybrid_retrieval import retrieve_hybrid_candidates
from rag.index_cache import (build_knowledge_index, cache_entry_count, get_knowledge_index,
                            get_cached_vector_store, rebuild_knowledge_index, reset_index_and_cache)
from embeddings.client import read_embedding_cache_identity
from rag.parent_child import ChunkingConfig, normalize_text, slice_spans, split_section
from rag.parent_retrieval import evidence_text, expand_parent_evidence
from rag.prompting import build_citations, render_rag_messages
from rag.quality import run_rag_quality_check
from rag.query_rewrite import rewrite_retrieval_query
from rag.retrieval import get_vector_store
from rag.reranker import RerankConfig, rerank_candidates
from test_document_loading import write_manifest, write_text_docx
from test_hybrid_rag import FakeEmbeddingClient
from test_product_tool_rag import ProductEmbeddingClient, ProductToolService


def section(text: str, *, identity: str = "policy", domain: str = "shipping", status: str = "active") -> KnowledgeSection:
    return KnowledgeSection(source_path=f"{identity}.md", document_title="政策",
                            section_index=1, section="规则", chunk_id=identity,
                            keywords=["物流", "时效"], effective_status=status,
                            text=normalize_text(text), metadata={"domain": domain})


def snapshot(text: str, **kwargs):
    config = ChunkingConfig()
    parents, children = split_section(section(text, **kwargs), config)
    return build_knowledge_index(children, parents, config), parents, children


def write_pages(path: Path, texts: list[str]) -> None:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
            {NameObject("/F1"): writer._add_object(font)})})
    with path.open("wb") as output:
        writer.write(output)


class ParentChildTests(unittest.TestCase):
    def setUp(self):
        reset_index_and_cache()
        self.addCleanup(reset_index_and_cache)

    def test_config_rejects_invalid_sizes_overlaps_and_types(self):
        for options in ({"parent_size": 0}, {"parent_overlap": 1600},
                        {"child_overlap": -1}, {"child_size": 1800}, {"parent_size": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ChunkingConfig(**options)
        with patch.dict("os.environ", {"AGENT_RAG_CHILD_SIZE": "300"}):
            self.assertEqual(ChunkingConfig.from_env().child_size, 300)

    def test_long_section_is_bounded_and_every_character_is_covered(self):
        text = normalize_text("\n\n".join(str(i) + "段" * 290 for i in range(18)))
        parents, children = split_section(section(text), ChunkingConfig())
        self.assertGreater(len(parents), 2)
        covered = set()
        for parent in parents:
            self.assertLessEqual(len(parent.text), 1600)
            covered.update(range(parent.metadata["section_start"], parent.metadata["section_end"]))
            own = [child for child in children if child.parent_id == parent.parent_id]
            child_coverage = set()
            for child in own:
                self.assertLessEqual(len(child.text), 420)
                self.assertEqual(child.text, parent.text[child.parent_start:child.parent_end])
                child_coverage.update(range(child.parent_start, child.parent_end))
            self.assertEqual(child_coverage, set(range(len(parent.text))))
        self.assertEqual(covered, set(range(len(text))))
        self.assertTrue(parents[0].text.endswith("\n\n"))
        build_knowledge_index(children, parents)

    def test_oversized_paragraph_falls_back_without_loss(self):
        _, parents, children = snapshot("超长段落" * 1000)
        self.assertGreater(len(parents), 1)
        self.assertTrue(all(len(child.text) <= 420 for child in children))
        self.assertEqual(parents[0].text[-160:], parents[1].text[:160])

    def test_short_section_keeps_legacy_id_and_explicit_parent_id(self):
        _, parents, children = snapshot("现货支付后发货。", identity="stable-shipping")
        self.assertEqual(children[0].chunk_id, "stable-shipping")
        self.assertEqual(children[0].parent_id, "stable-shipping-parent-1")
        self.assertEqual(parents[0].text, children[0].text)

    def test_parent_recovers_exception_outside_recalled_child(self):
        text = "现货支付后48小时内发货。" + "普通配送说明。" * 90 + "\n\n预售例外：以商品页预计发货时间为准。"
        index, parents, children = snapshot(text)
        hit = KnowledgeHit(chunk=children[0], score=0.9)
        self.assertNotIn("预售例外", hit.chunk.text)
        expansion = expand_parent_evidence([hit], index, "现货发货规则", ["shipping"])
        self.assertIn("预售例外", evidence_text(expansion.hits[0]))
        citation = build_citations(expansion.hits)[0]
        self.assertEqual(citation.snippet, evidence_text(expansion.hits[0]))
        self.assertEqual(citation.parent_id, parents[0].parent_id)
        self.assertEqual(citation.chunk_id, children[0].chunk_id)
        self.assertEqual(citation.index_version, index.version)
        messages = render_rag_messages("现货发货规则", IntentResult(intent="unknown", confidence=0.9,
                                      matched_keywords=[], source="rules", explanation="test"),
                                      rewrite_retrieval_query("现货发货规则", "unknown"), expansion.hits)
        self.assertIn(citation.snippet, messages[1]["content"])
        self.assertIn("[C1]", messages[1]["content"])

    def test_siblings_deduplicate_and_score_is_max_not_sum(self):
        index, _, children = snapshot("配送说明" * 260)
        hits = [KnowledgeHit(chunk=children[0], score=0.7),
                KnowledgeHit(chunk=children[1], score=0.9)]
        result = expand_parent_evidence(hits + hits, index, "物流", ["shipping"])
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].score, 0.9)
        self.assertEqual(result.hits[0].chunk.chunk_id, children[1].chunk_id)
        self.assertEqual(len(result.hits[0].matched_child_ids), 2)

    def test_equal_scores_preserve_upstream_order(self):
        p1, c1 = split_section(section("第一条", identity="z-policy"), ChunkingConfig())
        p2, c2 = split_section(section("第二条", identity="a-policy"), ChunkingConfig())
        index = build_knowledge_index(c1 + c2, p1 + p2)
        result = expand_parent_evidence([KnowledgeHit(chunk=chunk, score=0.9) for chunk in c1 + c2],
                                        index, "物流", ["shipping"], top_k=1)
        self.assertEqual(result.hits[0].chunk.chunk_id, "z-policy")

    def test_product_and_promotion_parents_survive_sibling_dominance(self):
        p1, c1 = split_section(section("商品说明" * 260, domain="product"), ChunkingConfig())
        p2, c2 = split_section(section("活动例外", identity="promotion", domain="promotion"), ChunkingConfig())
        index = build_knowledge_index(c1 + c2, p1 + p2)
        hits = [KnowledgeHit(chunk=c, score=0.95) for c in c1] + [KnowledgeHit(chunk=c2[0], score=0.7)]
        result = expand_parent_evidence(hits, index, "耳机", ["product", "promotion"],
                                        preferred_domains=("product", "promotion"), active_only=True)
        self.assertEqual([h.parent.metadata["domain"] for h in result.hits], ["product", "promotion"])

    def test_budget_drops_whole_parent_instead_of_truncating_exception(self):
        p1, c1 = split_section(section("物流规则" * 200), ChunkingConfig())
        p2, c2 = split_section(section("短物流规则", identity="short"), ChunkingConfig())
        index = build_knowledge_index(c1 + c2, p1 + p2)
        result = expand_parent_evidence([KnowledgeHit(chunk=c1[0], score=0.9),
                                         KnowledgeHit(chunk=c2[0], score=0.8)], index, "物流", ["shipping"],
                                        char_budget=50)
        self.assertEqual([h.chunk.chunk_id for h in result.hits], ["short"])
        self.assertEqual(result.trace["evidence_body_chars"], len(p2[0].text))
        self.assertEqual(result.trace["skipped"][0]["reason"], "evidence_char_budget")
        for options in ({"top_k": 0}, {"char_budget": -1}, {"top_k": True}):
            with self.assertRaises(ValueError):
                expand_parent_evidence([], index, "物流", ["shipping"], **options)

    def test_low_confidence_is_not_rescued_by_parent_content(self):
        index, _, children = snapshot("非常相关的完整发货规则")
        result = expand_parent_evidence([KnowledgeHit(chunk=children[0], score=0.49)], index, "发货", ["shipping"])
        self.assertFalse(result.hits)
        self.assertEqual(result.trace["skipped"][0]["reason"], "low_confidence")

    def test_filters_domain_status_and_allows_explicit_history(self):
        index, _, children = snapshot("历史物流政策", status="expired")
        hits = [KnowledgeHit(chunk=children[0], score=0.9)]
        self.assertFalse(expand_parent_evidence(hits, index, "物流", ["shipping"]).hits)
        self.assertTrue(expand_parent_evidence(hits, index, "历史物流", ["shipping"]).hits)
        self.assertFalse(expand_parent_evidence(hits, index, "历史物流", ["shipping"], active_only=True).hits)
        self.assertFalse(expand_parent_evidence(hits, index, "历史物流", ["promotion"]).hits)

    def test_missing_parent_stale_version_and_mismatch_fail_closed(self):
        index, _, children = snapshot("现货政策")
        hit = KnowledgeHit(chunk=children[0], score=0.9, index_version=index.version)
        cases = [index.model_copy(update={"parents_by_id": {}}),
                 index.model_copy(update={"version": "idx-other"})]
        for corrupt in cases:
            self.assertFalse(expand_parent_evidence([hit], corrupt, "物流", ["shipping"]).hits)
        forged = hit.model_copy(update={"chunk": hit.chunk.model_copy(update={"text": "伪造"})})
        self.assertFalse(expand_parent_evidence([forged], index, "物流", ["shipping"]).hits)

    def test_duplicate_orphan_offset_and_metadata_errors_reject_index(self):
        _, parents, children = snapshot("物流说明")
        invalid_sets = [(children * 2, parents), (children, parents * 2), (children, []),
                        ([children[0].model_copy(update={"parent_end": 1})], parents),
                        ([children[0].model_copy(update={"effective_status": "expired"})], parents),
                        ([children[0].model_copy(update={"metadata": {"domain": "product"}})], parents)]
        for chunks, parent_list in invalid_sets:
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                build_knowledge_index(chunks, parent_list)

    def test_fingerprint_covers_parent_only_changes_and_config(self):
        index, parents, children = snapshot("物流规则" * 230)
        # Change context outside all but the last child's range, with consistent mapping.
        changed_parents = [parents[0].model_copy(update={"text": parents[0].text[:-1] + "改"})]
        changed_children = [child.model_copy(update={
            "text": changed_parents[0].text[child.parent_start:child.parent_end]}) for child in children]
        changed = build_knowledge_index(changed_children, changed_parents)
        self.assertNotEqual(index.version, changed.version)
        config_only = build_knowledge_index(children, parents, ChunkingConfig(parent_overlap=150))
        self.assertNotEqual(index.version, config_only.version)

    def test_rebuild_failure_preserves_cache_and_inflight_old_snapshot(self):
        p1, c1 = split_section(section("物流规则", identity="old"), ChunkingConfig())
        old = rebuild_knowledge_index(c1, p1)
        client = FakeEmbeddingClient()
        outcome = retrieve_hybrid_candidates(rewrite_retrieval_query("物流时效", "unknown"),
                                             "unknown", embedding_client=client)
        self.assertGreater(cache_entry_count(), 0)
        old_count = cache_entry_count()
        with self.assertRaises(ValueError):
            rebuild_knowledge_index(c1, [])
        self.assertIs(get_knowledge_index(), old)
        self.assertEqual(cache_entry_count(), old_count)
        p2, c2 = split_section(section("新物流规则", identity="new"), ChunkingConfig())
        new = rebuild_knowledge_index(c2, p2)
        self.assertEqual(cache_entry_count(), 0)
        hit = outcome.candidates[0].model_copy(update={"score": 0.9})
        self.assertTrue(expand_parent_evidence([hit], old, "物流", ["shipping"]).hits)
        self.assertFalse(expand_parent_evidence([hit], new, "物流", ["shipping"]).hits)

    def test_embedding_and_commercial_rerank_receive_only_children(self):
        index, _, children = snapshot("物流说明" * 250)
        client = FakeEmbeddingClient()
        get_vector_store(client, index)
        self.assertEqual(len(client.seen_texts), len(children))
        bodies = []
        def handler(request):
            payload = json.loads(request.content)
            bodies.extend(payload["documents"])
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": 0.9} for i in range(len(children))]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            result = rerank_candidates("物流", [KnowledgeHit(chunk=c, score=0.8) for c in children],
                                       config=RerankConfig("test", "https://example.invalid", "mock", ""),
                                       http_client=http_client)
        self.assertEqual(result.mode, "commercial")
        self.assertEqual(bodies, [child.text for child in children])
        self.assertTrue(all(len(body) <= 420 for body in bodies))

    def test_inflight_old_embedding_cannot_repopulate_caches_after_rebuild(self):
        p1, c1 = split_section(section("物流规则", identity="old"), ChunkingConfig())
        old = rebuild_knowledge_index(c1, p1)
        p2, c2 = split_section(section("新物流规则", identity="new"), ChunkingConfig())
        class RebuildingClient(FakeEmbeddingClient):
            def embed_many(self, texts):
                rebuild_knowledge_index(c2, p2)
                return super().embed_many(texts)
        client = RebuildingClient()
        outcome = retrieve_hybrid_candidates(rewrite_retrieval_query("物流时效", "unknown"),
                                             "unknown", embedding_client=client)
        self.assertEqual(outcome.index.version, old.version)
        self.assertNotEqual(get_knowledge_index().version, old.version)
        self.assertIsNone(get_cached_vector_store(old.version, read_embedding_cache_identity(client)))
        self.assertEqual(cache_entry_count(), 0)
        hit = outcome.candidates[0].model_copy(update={"score": 0.9})
        self.assertTrue(expand_parent_evidence([hit], outcome.index, "物流", ["shipping"]).hits)


class LoaderAndAgentParentTests(unittest.TestCase):
    def setUp(self):
        reset_index_and_cache()
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.loader_patch = patch.object(knowledge_base, "KNOWLEDGE_DIR", self.directory)
        self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.addCleanup(reset_index_and_cache)

    def markdown(self, name, domain, body):
        (self.directory / f"{name}.md").write_text(
            f"---\ntitle: {name}\ndomain: {domain}\neffective_status: active\ntags: [物流, 时效, 耳机, 活动]\n---\n"
            f"## 规则\n<!-- chunk_id: {name} -->\n{body}", encoding="utf-8")

    def test_markdown_never_merges_h2_or_lifecycle_boundaries(self):
        self.markdown("shipping", "shipping", "当前政策\n\n## 历史规则\n"
                      "<!-- chunk_id: old; effective_status: expired -->\n历史政策")
        index = get_knowledge_index()
        self.assertEqual(len(index.parents_by_id), 2)
        self.assertEqual({p.effective_status for p in index.parents_by_id.values()}, {"active", "expired"})
        self.assertTrue(all(not ("当前政策" in p.text and "历史政策" in p.text)
                            for p in index.parents_by_id.values()))

    def test_real_long_txt_docx_and_markdown_loaders(self):
        text = "普通配送说明。" * 600 + "\n\n配送例外以商品页为准。"
        self.markdown("shipping", "shipping", text)
        txt = self.directory / "plain.txt"
        txt.write_text(text, encoding="utf-8")
        write_manifest(txt, document_id="plain")
        docx = self.directory / "word.docx"
        write_text_docx(docx, text)
        write_manifest(docx, document_id="word")
        index = get_knowledge_index()
        for name in ("shipping.md", "plain.txt", "word.docx"):
            parents = [p for p in index.parents_by_id.values() if p.source_path == name]
            self.assertGreater(len(parents), 1)
            self.assertTrue(any("配送例外" in p.text for p in parents))

    def test_cross_page_pdf_and_citation_have_actual_parent_page_range(self):
        path = self.directory / "pages.pdf"
        write_pages(path, ["Shipping rule. " + "A" * 350, "Exception: preorders follow the product page."])
        write_manifest(path)
        index = get_knowledge_index()
        children = list(index.chunks_by_id.values())
        first = children[0]
        result = expand_parent_evidence([KnowledgeHit(chunk=first, score=0.9)], index, "物流", ["shipping"])
        citation = build_citations(result.hits)[0]
        self.assertEqual((citation.page_start, citation.page_end), (1, 2))
        self.assertEqual(citation.section, "第 1–2 页")
        self.assertIn("Exception", citation.snippet)
        self.assertEqual(first.source_spans, slice_spans(result.hits[0].parent.source_spans,
                                                        first.parent_start, first.parent_end))

    def test_long_pdf_parents_do_not_all_claim_entire_document_page_range(self):
        path = self.directory / "long.pdf"
        write_pages(path, ["A" * 1100, "B" * 1100, "C" * 1100])
        write_manifest(path)
        index = get_knowledge_index()
        parents = list(index.parents_by_id.values())
        self.assertGreater(len(parents), 1)
        self.assertEqual(parents[0].metadata["page_start"], 1)
        self.assertLess(parents[0].metadata["page_end"], 3)
        self.assertGreater(parents[-1].metadata["page_start"], 1)

    def test_normal_chat_prompt_and_offline_quality_include_parent_exception(self):
        self.markdown("shipping", "shipping", "现货支付后48小时内发货。" + "普通配送说明。" * 90 +
                      "\n\n预售例外：以商品页预计发货时间为准。")
        payloads = []
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "预售以商品页为准。[C1]"}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            agent = CustomerServiceAgent(embedding_client=FakeEmbeddingClient(), answer_api_key="test",
                                         answer_http_client=client, answer_base_url="https://example.invalid")
            result = agent.chat(ChatRequest(session_id="parent-evidence", runtime_user_id="PRIVATE-ID",
                                            user_message="现货物流时效是什么？"))
        self.assertTrue(result.citations)
        self.assertIn("预售例外", result.citations[0].snippet)
        self.assertIn("预售例外", str(payloads))
        self.assertNotIn("PRIVATE-ID", str(payloads))
        self.assertEqual(result.session_state["rag"]["parent_expansion"]["selected_parent_ids"],
                         [result.citations[0].parent_id])
        index = get_knowledge_index()
        expected = list(index.chunks_by_id)[0]
        summary = run_rag_quality_check([RagQualityCase(case_id="shipping-exception", question="现货物流时效",
                                       expected_chunk_ids=[expected], required_evidence_terms=["48小时", "预售例外"])],
                                       embedding_client=FakeEmbeddingClient())
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.results[0].required_evidence_coverage, 1.0)

    def test_product_joint_and_after_sale_policy_use_parent_citations(self):
        self.markdown("product", "product", "降噪耳机适合通勤。" + "耳机续航说明。" * 90 + "\n\n商品限制：并非专业隔音设备。")
        self.markdown("promotion", "promotion", "当前耳机活动允许优惠券。" + "活动规则说明。" * 85 + "\n\n活动例外：最终以结算页为准。")
        self.markdown("after-sale", "after_sale", "七天无理由退货。" + "售后规则说明。" * 85 + "\n\n售后例外：特殊商品需人工核验。")
        agent = CustomerServiceAgent(embedding_client=ProductEmbeddingClient(), answer_api_key="",
                                     tool_calling_service=ProductToolService())
        result = agent.chat(ChatRequest(session_id="joint-parent", runtime_user_id="U1",
                                        user_message="耳机现在价格库存怎样，适不适合通勤，有什么活动？"))
        self.assertEqual(len(result.citations), 2)
        self.assertTrue(all(c.parent_id for c in result.citations))
        self.assertIn("商品限制", result.answer)
        self.assertIn("活动例外", result.answer)
        request = ChatRequest(session_id="after-parent", runtime_user_id="U1", user_message="七天无理由退货售后规则")
        citations, trace = agent._retrieve_after_sale_policy(request, IntentResult(
            intent="refund_request", confidence=1.0, matched_keywords=[], source="rules", explanation="test"))
        self.assertTrue(citations)
        self.assertIn("售后例外", citations[0].snippet)
        self.assertTrue(trace["parent_expansion"]["selected_parent_ids"])

    def test_config_change_rejects_old_hits_even_when_child_text_is_identical(self):
        self.markdown("shipping", "shipping", "短物流规则")
        old = get_knowledge_index()
        child = next(iter(old.chunks_by_id.values()))
        hit = KnowledgeHit(chunk=child, score=0.9, index_version=old.version)
        new = rebuild_knowledge_index(config=ChunkingConfig(parent_overlap=150))
        self.assertEqual(new.chunks_by_id[child.chunk_id], child)
        self.assertFalse(expand_parent_evidence([hit], new, "物流", ["shipping"]).hits)

    def test_broken_knowledge_is_not_reparsed_outside_fallback_and_general_chat(self):
        path = self.directory / "shipping.txt"
        path.write_text("物流规则", encoding="utf-8")  # deliberately missing manifest
        agent = CustomerServiceAgent(embedding_client=FakeEmbeddingClient(), answer_api_key="")
        response = agent.chat(ChatRequest(session_id="bad-corpus", runtime_user_id="U1", user_message="物流时效是什么？"))
        self.assertFalse(response.citations)
        self.assertEqual(response.session_state["rag"]["answer_path"], "retrieval_unavailable")
        with patch.object(knowledge_base, "load_source_documents", side_effect=AssertionError("unexpected parse")):
            general = agent.chat(ChatRequest(session_id="no-rag", runtime_user_id="U1", user_message="你好"))
        self.assertFalse(general.citations)


if __name__ == "__main__":
    unittest.main()
