"""Offline regression tests for the lesson-15 Hybrid RAG increment."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import ChatRequest  # noqa: E402
from embeddings.client import EmbeddingClient  # noqa: E402
from rag.hybrid_retrieval import (  # noqa: E402
    retrieve_hybrid_candidates,
    retrieve_keyword_candidates,
)
from rag.index_cache import (  # noqa: E402
    cache_entry_count,
    get_knowledge_index,
    rebuild_knowledge_index,
    reset_index_and_cache,
)
from rag.knowledge_base import load_knowledge_chunks  # noqa: E402
from rag.planning import build_retrieval_plan  # noqa: E402
from rag.quality import run_rag_quality_check  # noqa: E402
from rag.query_rewrite import rewrite_retrieval_query  # noqa: E402


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic semantic buckets used without a model provider."""

    def __init__(self) -> None:
        super().__init__(api_key="test-key", base_url="https://example.invalid/v1")
        self.seen_texts: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        buckets = [
            ["优惠", "活动", "会员价", "优惠券", "满减", "音频节"],
            ["售后", "退货", "退款", "无理由", "包装", "配件", "赠品"],
            ["物流", "快递", "发货", "配送", "运单"],
            ["商品", "耳机", "充电器", "音箱", "推荐"],
        ]
        return [
            1.0 if any(term in text for term in bucket) else 0.0 for bucket in buckets
        ]

    def embed(self, text: str) -> list[float]:
        self.seen_texts.append(text)
        return self._vector(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        return [self._vector(text) for text in texts]


class HybridRagTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_index_and_cache()

    def tearDown(self) -> None:
        reset_index_and_cache()

    def test_plan_preserves_original_query_and_routes_promotion(self) -> None:
        rewrite = rewrite_retrieval_query(
            "耳麦会员价还能叠券吗？",
            "promotion_consult",
        )

        plan = build_retrieval_plan(rewrite, "promotion_consult")

        self.assertEqual(plan.original_query, "耳麦会员价还能叠券吗？")
        self.assertEqual(plan.scene, "promotion")
        self.assertIn("promotion", plan.allowed_domains)
        self.assertIn("优惠券", plan.keyword_terms)

    def test_long_tail_after_sale_terms_enter_keyword_route(self) -> None:
        rewrite = rewrite_retrieval_query(
            "退货时赠品少一根线怎么办？",
            "refund_request",
        )
        plan = build_retrieval_plan(rewrite, "refund_request")

        hits = retrieve_keyword_candidates(plan)

        self.assertEqual(plan.scene, "after_sale")
        self.assertTrue(hits)
        self.assertIn("keyword", hits[0].retrieval_sources)
        self.assertTrue({"赠品", "配件"} & set(hits[0].matched_keywords))

    def test_fixed_quality_set_passes_through_hybrid_chain(self) -> None:
        summary = run_rag_quality_check(embedding_client=FakeEmbeddingClient())

        self.assertEqual(summary.total_cases, 3)
        self.assertEqual(summary.passed_cases, 3)

    def test_complaint_intent_wins_over_refund_noun(self) -> None:
        rewrite = rewrite_retrieval_query("退款没处理，我要投诉", "complaint")

        plan = build_retrieval_plan(rewrite, "complaint")

        self.assertEqual(plan.scene, "complaint")
        self.assertIn("complaint", plan.allowed_domains)

    def test_agent_exposes_three_routes_without_leaking_runtime_identity(self) -> None:
        embedding_client = FakeEmbeddingClient()
        agent = CustomerServiceAgent(
            embedding_client=embedding_client,
            answer_api_key="",
        )
        secret_values = ["USER-PRIVATE-9001", "私密昵称", "diamond-secret"]

        response = agent.chat(
            ChatRequest(
                session_id="hybrid-test",
                runtime_user_id=secret_values[0],
                runtime_nickname=secret_values[1],
                runtime_member_level=secret_values[2],
                runtime_risk_level="low",
                user_message="金卡会员买降噪耳机，会员价还能叠加优惠券吗？",
            )
        )

        rag_state = response.session_state["rag"]
        self.assertEqual(rag_state["plan"]["scene"], "promotion")
        self.assertTrue(rag_state["original_vector_chunk_ids"])
        self.assertTrue(rag_state["rewritten_vector_chunk_ids"])
        self.assertTrue(rag_state["keyword_chunk_ids"])
        self.assertTrue(response.citations)
        external_embedding_text = "\n".join(embedding_client.seen_texts)
        for secret in secret_values:
            self.assertNotIn(secret, external_embedding_text)

    def test_general_chat_bypasses_all_retrieval_routes(self) -> None:
        embedding_client = FakeEmbeddingClient()
        agent = CustomerServiceAgent(embedding_client=embedding_client)

        response = agent.chat(
            ChatRequest(
                session_id="general-chat-test",
                runtime_user_id="U1",
                user_message="你好",
            )
        )

        self.assertEqual(response.intent, "general_chat")
        self.assertEqual(response.session_state["rag"]["candidate_count"], 0)
        self.assertEqual(embedding_client.seen_texts, [])

    def test_stable_hybrid_candidates_hit_cache_on_second_query(self) -> None:
        embedding_client = FakeEmbeddingClient()
        rewrite = rewrite_retrieval_query(
            "耳机会员价还能叠加优惠券吗？",
            "promotion_consult",
        )

        first = retrieve_hybrid_candidates(
            rewrite,
            "promotion_consult",
            embedding_client=embedding_client,
        )
        calls_after_first = len(embedding_client.seen_texts)
        second = retrieve_hybrid_candidates(
            rewrite,
            "promotion_consult",
            embedding_client=embedding_client,
        )

        self.assertFalse(first.cache["cache_hit"])
        self.assertTrue(second.cache["cache_hit"])
        self.assertEqual(first.index.version, second.index.version)
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(len(embedding_client.seen_texts), calls_after_first)

    def test_realtime_business_query_is_never_put_in_retrieval_cache(self) -> None:
        embedding_client = FakeEmbeddingClient()
        rewrite = rewrite_retrieval_query("我的订单到哪了？", "order_query")

        first = retrieve_hybrid_candidates(
            rewrite,
            "order_query",
            embedding_client=embedding_client,
        )
        second = retrieve_hybrid_candidates(
            rewrite,
            "order_query",
            embedding_client=embedding_client,
        )

        self.assertFalse(first.cache["cacheable"])
        self.assertFalse(first.cache["cache_hit"])
        self.assertFalse(second.cache["cache_hit"])
        self.assertIsNone(first.cache["cache_key"])

    def test_rebuild_changes_version_and_invalidates_dependent_caches(self) -> None:
        embedding_client = FakeEmbeddingClient()
        rewrite = rewrite_retrieval_query("耳机活动有什么优惠？", "promotion_consult")
        retrieve_hybrid_candidates(
            rewrite,
            "promotion_consult",
            embedding_client=embedding_client,
        )
        original_index = get_knowledge_index()
        self.assertGreater(cache_entry_count(), 0)
        changed_chunks = load_knowledge_chunks()
        changed_chunks[0] = changed_chunks[0].model_copy(
            update={"text": changed_chunks[0].text + " 测试版知识变更。"}
        )

        rebuilt = rebuild_knowledge_index(changed_chunks)

        self.assertNotEqual(original_index.version, rebuilt.version)
        self.assertEqual(cache_entry_count(), 0)

    def test_embedding_model_identity_uses_a_separate_cache_namespace(self) -> None:
        embedding_client = FakeEmbeddingClient()
        rewrite = rewrite_retrieval_query("耳机活动有什么优惠？", "promotion_consult")
        first = retrieve_hybrid_candidates(
            rewrite,
            "promotion_consult",
            embedding_client=embedding_client,
        )
        embedding_client.model = "fake-embedding-v2"

        second = retrieve_hybrid_candidates(
            rewrite,
            "promotion_consult",
            embedding_client=embedding_client,
        )

        self.assertFalse(first.cache["cache_hit"])
        self.assertFalse(second.cache["cache_hit"])
        self.assertNotEqual(
            first.cache["embedding_identity_hash"],
            second.cache["embedding_identity_hash"],
        )

    def test_agent_realtime_gap_has_no_citations_or_model_answer(self) -> None:
        agent = CustomerServiceAgent(
            embedding_client=FakeEmbeddingClient(),
            answer_api_key="test-key-that-must-not-be-used",
        )

        response = agent.chat(
            ChatRequest(
                session_id="realtime-gap-test",
                runtime_user_id="U1",
                user_message="我的订单到哪了？",
            )
        )

        self.assertEqual(response.citations, [])
        self.assertTrue(response.session_state["rag"]["realtime_gap"])
        self.assertFalse(response.session_state["rag"]["cache"]["cacheable"])
        self.assertEqual(
            response.session_state["rag"]["answer_path"],
            "realtime_business_tool_required",
        )
        self.assertFalse(response.session_state["model_answer"]["used_model"])

    def test_cached_candidates_do_not_cache_final_model_answer(self) -> None:
        model_requests: list[httpx.Request] = []

        def answer_handler(request: httpx.Request) -> httpx.Response:
            model_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "活动规则以结算页为准。[C1]"}}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                },
            )

        answer_client = httpx.Client(transport=httpx.MockTransport(answer_handler))
        agent = CustomerServiceAgent(
            embedding_client=FakeEmbeddingClient(),
            answer_http_client=answer_client,
            answer_api_key="test-key",
            answer_base_url="https://example.invalid/v1",
        )
        request = ChatRequest(
            session_id="answer-cache-boundary",
            runtime_user_id="PRIVATE-USER-ID",
            runtime_nickname="PRIVATE-NICKNAME",
            user_message="耳机会员价还能叠加优惠券吗？",
        )
        try:
            first = agent.chat(request)
            second = agent.chat(request)
        finally:
            answer_client.close()

        self.assertFalse(first.session_state["rag"]["cache"]["cache_hit"])
        self.assertTrue(second.session_state["rag"]["cache"]["cache_hit"])
        self.assertEqual(len(model_requests), 2)
        request_bodies = "\n".join(
            request.content.decode("utf-8") for request in model_requests
        )
        self.assertNotIn("PRIVATE-USER-ID", request_bodies)
        self.assertNotIn("PRIVATE-NICKNAME", request_bodies)


if __name__ == "__main__":
    unittest.main()
