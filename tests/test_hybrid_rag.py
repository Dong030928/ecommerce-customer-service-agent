"""Offline regression tests for the lesson-15 Hybrid RAG increment."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import ChatRequest  # noqa: E402
from embeddings.client import EmbeddingClient  # noqa: E402
from rag.hybrid_retrieval import retrieve_keyword_candidates  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
