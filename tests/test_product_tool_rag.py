"""Offline tests for lesson 22 product Tool + RAG joint answers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import (  # noqa: E402
    ChatRequest,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from embeddings.client import EmbeddingClient  # noqa: E402
from rag.index_cache import reset_index_and_cache  # noqa: E402
from tools.tool_calling import ToolCallingOutcome  # noqa: E402


class ProductEmbeddingClient(EmbeddingClient):
    """Deterministic product/promotion vectors with input capture."""

    def __init__(self) -> None:
        super().__init__(api_key="test-key", base_url="https://example.invalid/v1")
        self.seen_texts: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        buckets = [
            ["耳机", "降噪", "通勤", "差旅", "续航", "推荐", "商品"],
            ["活动", "优惠", "会员价", "优惠券", "叠加", "结算页", "音频节"],
            ["退款", "退货", "售后"],
        ]
        return [
            1.0 if any(term in text for term in bucket) else 0.0
            for bucket in buckets
        ]

    def embed(self, text: str) -> list[float]:
        self.seen_texts.append(text)
        return self._vector(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        return [self._vector(text) for text in texts]


class ProductToolService:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *args: object, **kwargs: object) -> ToolCallingOutcome:
        del args, kwargs
        self.calls += 1
        observation = ToolObservation(
            tool_name="get_product_inventory",
            status="success",
            summary=(
                "降噪蓝牙耳机当前价 299.00 元，库存 18 件，"
                "活动价 259.00 元，当前活动为春季音频节会员价。"
            ),
            facts={
                "sku": "SKU-AUD-101",
                "name": "降噪蓝牙耳机",
                "current_price": "299.00",
                "promotion_price": "259.00",
                "inventory": 18,
                "active": True,
                "promotion_name": "春季音频节会员价",
                "promotion_summary": "金卡会员活动价",
            },
            omitted_fields=["product.description", "product.promotion"],
            next_action="answer_user",
        )
        return ToolCallingOutcome(
            answer="工具单路回答不应成为联合回答的最终内容。",
            tool_calls=[
                ToolCallRecord(
                    action=ToolAction(
                        tool_name="get_product_inventory",
                        arguments={"sku": "耳机"},
                        reason="查询实时商品价格和库存。",
                    ),
                    observation=observation,
                )
            ],
            state={
                "create_agent": True,
                "answer_source": "compressed_observation_fallback",
            },
            used_model=True,
            model_name="test-tool-model",
        )


class ProductToolRagTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_index_and_cache()
        self.embedding_client = ProductEmbeddingClient()
        self.tool_service = ProductToolService()
        self.agent = CustomerServiceAgent(
            embedding_client=self.embedding_client,
            answer_api_key="",
            tool_calling_service=self.tool_service,
        )

    def tearDown(self) -> None:
        reset_index_and_cache()

    @staticmethod
    def request(message: str) -> ChatRequest:
        return ChatRequest(
            session_id="product-tool-rag-test",
            runtime_user_id="PRIVATE-RUNTIME-USER",
            runtime_nickname="PRIVATE-NICKNAME",
            user_message=message,
        )

    def test_mixed_product_question_returns_tool_calls_and_citations(self) -> None:
        response = self.agent.chat(
            self.request("我通勤想买降噪耳机，现在有库存吗，活动怎么算？")
        )

        self.assertEqual(response.intent, "product_consult")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertTrue(response.citations)
        self.assertEqual(
            response.session_state["tool_rag"]["answer_sources"],
            ["tool", "rag"],
        )
        self.assertTrue(response.session_state["tool_rag"]["joint_answer_complete"])
        self.assertEqual(
            response.session_state["tool_rag"]["source_boundary"],
            {
                "current_price_inventory": "tool",
                "product_and_promotion_knowledge": "rag",
            },
        )
        self.assertIn("299.00", response.answer)
        self.assertIn("259.00", response.answer)
        self.assertIn("[C1]", response.answer)
        citation_ids = {citation.chunk_id for citation in response.citations}
        self.assertIn("product-anc-headphone", citation_ids)
        self.assertIn("promotion-current-audio-offer", citation_ids)

    def test_runtime_identity_never_enters_joint_model_or_embedding_context(self) -> None:
        model_payloads: list[dict] = []

        def handler(http_request: httpx.Request) -> httpx.Response:
            model_payloads.append(json.loads(http_request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "工具显示库存 18 件、活动价 259 元；规则以结算页为准。[C1]"
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            agent = CustomerServiceAgent(
                embedding_client=self.embedding_client,
                answer_http_client=http_client,
                answer_api_key="test-key",
                answer_base_url="https://model.invalid/v1",
                tool_calling_service=self.tool_service,
            )
            response = agent.chat(
                self.request("我通勤想买降噪耳机，现在有库存吗，活动怎么算？")
            )
        finally:
            http_client.close()

        serialized_answer = response.answer
        embedding_context = "\n".join(self.embedding_client.seen_texts)
        model_context = json.dumps(model_payloads, ensure_ascii=False)
        for secret in ["PRIVATE-RUNTIME-USER", "PRIVATE-NICKNAME"]:
            self.assertNotIn(secret, serialized_answer)
            self.assertNotIn(secret, embedding_context)
            self.assertNotIn(secret, model_context)
        self.assertIn("TOOL_OBSERVATIONS", model_context)
        self.assertIn("RAG_EVIDENCE", model_context)
        self.assertFalse(
            response.session_state["tool_calling"]["raw_tool_result_exposed"]
        )

    def test_realtime_only_product_question_keeps_the_tool_only_route(self) -> None:
        response = self.agent.chat(
            self.request("SKU-AUD-101 现在还有库存吗？")
        )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.citations, [])
        self.assertNotIn("tool_rag", response.session_state)
        self.assertEqual(
            response.session_state["rag"]["status"],
            "skipped_realtime_tool_route",
        )

    def test_stable_product_question_keeps_the_rag_only_route(self) -> None:
        response = self.agent.chat(
            self.request("降噪耳机适合通勤和差旅吗？")
        )

        self.assertEqual(response.tool_calls, [])
        self.assertTrue(response.citations)
        self.assertEqual(self.tool_service.calls, 0)
        self.assertNotIn("tool_rag", response.session_state)

    def test_joint_model_answer_without_citation_marker_falls_back(self) -> None:
        def handler(http_request: httpx.Request) -> httpx.Response:
            del http_request
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "这是一个没有引用标记的模型回答。"}}
                    ]
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            agent = CustomerServiceAgent(
                embedding_client=self.embedding_client,
                answer_http_client=http_client,
                answer_api_key="test-key",
                answer_base_url="https://model.invalid/v1",
                tool_calling_service=self.tool_service,
            )
            response = agent.chat(
                self.request("我通勤想买降噪耳机，现在有库存吗，活动怎么算？")
            )
        finally:
            http_client.close()

        self.assertTrue(response.degraded)
        self.assertIn("[C1]", response.answer)
        self.assertEqual(
            response.session_state["model_answer"]["fallback_reason"],
            "citation_marker_missing",
        )


if __name__ == "__main__":
    unittest.main()
