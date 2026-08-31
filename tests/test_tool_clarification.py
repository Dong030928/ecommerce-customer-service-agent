"""Offline tests for lesson-19 structured tool clarification."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import ChatRequest, ToolAction, ToolCallRecord, ToolObservation  # noqa: E402
from models.clarification_planner import plan_clarification_with_model  # noqa: E402
from tools.planning import (  # noqa: E402
    apply_model_clarification_draft,
    build_clarification_plan,
    post_tool_clarification,
)
from tools.tool_runtime import ToolRuntime  # noqa: E402


def order_context() -> dict:
    return {
        "currentUserOrders": [
            {
                "orderNo": "SO20260501000000001-a1000001",
                "status": "SHIPPED",
                "createdAt": "2026-05-01T10:00:00",
                "itemSummary": ["降噪蓝牙耳机"],
            },
            {
                "orderNo": "SO20260502000000002-a1000001",
                "status": "PAID",
                "createdAt": "2026-05-02T10:00:00",
                "itemSummary": ["桌面音箱"],
            },
            {
                "orderNo": "SO20260601000000003-a1000001",
                "status": "DELIVERED",
                "createdAt": "2026-06-01T10:00:00",
                "itemSummary": ["充电器"],
            },
        ],
        "currentUserOrdersTruncated": False,
    }


class ToolClarificationTests(unittest.TestCase):
    def test_missing_order_id_returns_structured_candidates_without_tool_call(self) -> None:
        agent = CustomerServiceAgent()
        response = agent.chat(
            ChatRequest(
                session_id="clarification-test",
                runtime_user_id="PRIVATE-USER-ID",
                user_message="我的物流到哪了？",
                runtime_context=order_context(),
            )
        )

        self.assertIsNotNone(response.clarification)
        self.assertEqual(response.clarification.clarification_field, "order_id")
        self.assertEqual(len(response.clarification.candidates), 3)
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(
            response.session_state["tool_calling"]["clarification_stage"],
            "pre_tool",
        )
        self.assertEqual(response.next_action, "ask_clarification")
        self.assertNotIn("PRIVATE-USER-ID", response.answer)

    def test_model_draft_cannot_inject_identity_or_invent_order(self) -> None:
        request = ChatRequest(
            session_id="draft-guard",
            runtime_user_id="PRIVATE-USER-ID",
            user_message="我的物流到哪了？",
        )
        authoritative = build_clarification_plan(request, "order_query")
        validated = apply_model_clarification_draft(
            authoritative,
            {
                "tool_name": "get_order_logistics",
                "known_arguments": {
                    "order_id": "SO-ATTACKER",
                    "user_id": "ATTACKER",
                },
                "missing_required": [],
                "clarification_question": "请提供订单号。",
                "confidence": 0.88,
                "reason": "test",
            },
            model_name="test-model",
        )

        self.assertEqual(validated.known_arguments, {})
        self.assertEqual(validated.missing_required, ["order_id"])
        self.assertEqual(validated.source, "model")

    def test_model_draft_cannot_replace_clarification_with_unrelated_request(self) -> None:
        request = ChatRequest(
            session_id="draft-question-guard",
            runtime_user_id="PRIVATE-USER-ID",
            user_message="我的物流到哪了？",
        )
        authoritative = build_clarification_plan(request, "order_query")
        validated = apply_model_clarification_draft(
            authoritative,
            {
                "tool_name": "get_order_logistics",
                "clarification_question": "请提供订单号和支付验证码。",
                "confidence": 1,
                "reason": "test",
            },
            model_name="test-model",
        )

        self.assertIsNone(validated.clarification_question)

    def test_unique_trusted_order_enriches_plan_without_model_guess(self) -> None:
        context = order_context()
        context["currentUserOrders"] = context["currentUserOrders"][:1]
        request = ChatRequest(
            session_id="unique-order",
            runtime_user_id="PRIVATE-USER-ID",
            user_message="我的物流到哪了？",
            runtime_context=context,
        )
        plan = build_clarification_plan(request, "order_query")

        self.assertEqual(
            plan.known_arguments["order_id"],
            "SO20260501000000001-a1000001",
        )
        self.assertEqual(plan.missing_required, [])
        self.assertEqual(plan.source, "backend_guard")

    def test_month_search_keeps_multiple_candidates_for_user_confirmation(self) -> None:
        request = ChatRequest(
            session_id="month-orders",
            runtime_user_id="PRIVATE-USER-ID",
            user_message="查一下 5 月的订单到哪了",
            runtime_context=order_context(),
        )
        plan = build_clarification_plan(request, "order_query")
        action = ToolAction(
            tool_name=str(plan.tool_name),
            arguments=plan.known_arguments,
            reason=plan.reason,
        )
        observation = ToolRuntime().execute(action, request)
        clarification = post_tool_clarification(
            [ToolCallRecord(action=action, observation=observation)]
        )

        self.assertEqual(plan.tool_name, "search_current_user_orders")
        self.assertEqual(observation.status, "success")
        self.assertEqual(len(observation.data["candidate_orders"]), 2)
        self.assertIsNotNone(clarification)
        self.assertEqual(len(clarification.candidates), 2)

    def test_model_planner_sees_no_runtime_identity_and_backend_keeps_arguments(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"tool_name":"get_order_logistics",'
                                    '"known_arguments":{"user_id":"ATTACKER"},'
                                    '"clarification_question":"请确认订单号。",'
                                    '"confidence":0.9,"reason":"缺少订单号"}'
                                )
                            }
                        }
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        request = ChatRequest(
            session_id="model-clarification",
            runtime_user_id="PRIVATE-USER-ID",
            runtime_nickname="PRIVATE-NICKNAME",
            user_message="我的物流到哪了？",
        )
        authoritative = build_clarification_plan(request, "order_query")
        try:
            result = plan_clarification_with_model(
                request,
                authoritative,
                http_client=client,
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
            )
        finally:
            client.close()

        sent = requests[0].content.decode("utf-8")
        self.assertNotIn("PRIVATE-USER-ID", sent)
        self.assertNotIn("PRIVATE-NICKNAME", sent)
        self.assertEqual(result.known_arguments, {})
        self.assertEqual(result.missing_required, ["order_id"])
        self.assertEqual(result.source, "model")

    def test_ambiguous_product_observation_becomes_post_tool_clarification(self) -> None:
        record = ToolCallRecord(
            action=ToolAction(
                tool_name="get_product_inventory",
                arguments={"sku": "耳机"},
                reason="test",
            ),
            observation=ToolObservation(
                tool_name="get_product_inventory",
                status="error",
                summary="没有唯一匹配的商品。",
                data={
                    "candidates": [
                        {"code": "SKU-AUD-101", "name": "降噪蓝牙耳机"},
                        {"code": "SKU-AUD-102", "name": "运动蓝牙耳机"},
                    ]
                },
                error_code="product_ambiguous",
            ),
        )
        clarification = post_tool_clarification([record])

        self.assertIsNotNone(clarification)
        self.assertEqual(clarification.clarification_field, "sku")
        self.assertEqual(len(clarification.candidates), 2)


if __name__ == "__main__":
    unittest.main()
