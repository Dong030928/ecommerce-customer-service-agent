"""Offline regression tests for TaskPlanner and validated RoutePlan output."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import (  # noqa: E402
    CustomerServiceAgent,
    plan_intent_by_rules,
)
from api.schemas import (  # noqa: E402
    ChatRequest,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from planner.task_planner import TaskPlanner  # noqa: E402
from models.task_planner_client import TaskPlannerModelClient  # noqa: E402
from tools.tool_calling import ToolCallingOutcome  # noqa: E402


class DraftModelClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, list[str]]] = []

    def plan(self, message: str, candidates: list[object]) -> dict[str, object]:
        self.calls.append(
            (
                message,
                [str(getattr(candidate, "name")) for candidate in candidates],
            )
        )
        return self.payload


class CapturingToolService:
    def __init__(self) -> None:
        self.allowed_tool_names: list[str] | None = None

    def run(self, *args: object, **kwargs: object) -> ToolCallingOutcome:
        allowed = kwargs.get("allowed_tool_names")
        self.allowed_tool_names = list(allowed) if isinstance(allowed, list) else None
        record = ToolCallRecord(
            action=ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": "SO20260420103000001-a1000001"},
                reason="RoutePlan 收窄到订单状态工具。",
            ),
            observation=ToolObservation(
                tool_name="get_order_status",
                status="success",
                summary="订单状态为 SHIPPED。",
                facts={"order_status": "SHIPPED"},
            ),
        )
        return ToolCallingOutcome(
            answer=record.observation.summary,
            tool_calls=[record],
            state={"answer_source": "compressed_observation_fallback"},
            used_model=True,
            model_name="planner-test-tool-model",
        )


class TaskPlannerTests(unittest.TestCase):
    def _request(self, message: str) -> ChatRequest:
        return ChatRequest(
            session_id="task-planner-test",
            runtime_user_id="PRIVATE-RUNTIME-USER",
            user_message=message,
            runtime_context={"currentPage": "order-detail"},
        )

    def _plan(self, message: str, planner: TaskPlanner | None = None):
        request = self._request(message)
        rule_result = plan_intent_by_rules(message)
        return (planner or TaskPlanner()).plan(request, rule_result)

    def test_rule_planner_routes_rag_tool_mixed_and_general_paths(self) -> None:
        cases = [
            ("你好", "general", []),
            ("退款规则是什么", "rag", []),
            (
                "查询订单 SO20260420103000001-a1000001 的状态",
                "tool",
                ["get_order_status"],
            ),
            (
                "SKU-AUD-101 现在有库存吗，适合通勤吗？",
                "tool_rag",
                ["get_product_inventory"],
            ),
        ]

        for message, expected_route, expected_tools in cases:
            with self.subTest(message=message):
                plan, trace = self._plan(message)
                self.assertEqual(plan.execution_route, expected_route)
                self.assertEqual(plan.required_tools, expected_tools)
                self.assertEqual(trace.constrained_required_tools, expected_tools)
                self.assertFalse(trace.model_consulted)

    def test_high_risk_write_is_forced_to_unexecuted_workflow_route(self) -> None:
        model = DraftModelClient(
            {
                "intent": "general_chat",
                "needs_rag": False,
                "needs_business_tools": False,
                "confidence": 0.99,
                "risk_level": "low",
                "requires_workflow": False,
            }
        )
        plan, trace = self._plan(
            "请把订单 SO20260420103000001-a1000001 直接退款并赔付",
            TaskPlanner(model),
        )

        self.assertEqual(plan.execution_route, "workflow")
        self.assertTrue(plan.requires_workflow)
        self.assertEqual(plan.risk_level, "high")
        self.assertEqual(plan.required_tools, [])
        self.assertFalse(plan.needs_business_tools)
        self.assertEqual(model.calls, [])
        self.assertIn("不在轻路径执行", trace.public_reason)

    def test_low_confidence_model_draft_is_allow_list_constrained(self) -> None:
        model = DraftModelClient(
            {
                "intent": "product_consult",
                "needs_rag": True,
                "needs_business_tools": True,
                "rag_query": "耳机 商品知识",
                "confidence": 0.82,
                "intents": ["product_consult", "delete_everything"],
                "entity_refs": ["sku", "private_user_id"],
                "required_context": ["runtime_context", "password"],
                "required_tools": ["delete_order", "get_product_inventory"],
                "knowledge_domains": ["product", "private"],
                "has_realtime_fact": True,
                "risk_level": "low",
                "requires_workflow": False,
                "fallback_policy": "tool_first_then_policy_caveat",
            }
        )
        plan, trace = self._plan("帮我看看这个", TaskPlanner(model))

        self.assertEqual(plan.source, "classifier")
        self.assertEqual(plan.execution_route, "tool_rag")
        self.assertEqual(plan.required_tools, ["get_product_inventory"])
        self.assertEqual(plan.entity_refs, ["sku"])
        self.assertEqual(plan.required_context, ["runtime_context"])
        self.assertEqual(plan.knowledge_domains, ["product"])
        self.assertEqual(plan.intents, ["product_consult"])
        self.assertTrue(trace.model_consulted)
        self.assertNotIn("delete_order", trace.constrained_required_tools)
        self.assertNotIn("PRIVATE-RUNTIME-USER", str(model.calls))

    def test_real_planner_client_receives_no_runtime_identity(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"intent":"promotion_consult",'
                                    '"needs_rag":true,'
                                    '"needs_business_tools":false,'
                                    '"rag_query":"优惠规则",'
                                    '"confidence":0.81,'
                                    '"required_tools":[],'
                                    '"knowledge_domains":["promotion"],'
                                    '"has_realtime_fact":false,'
                                    '"risk_level":"low",'
                                    '"requires_workflow":false}'
                                )
                            }
                        }
                    ]
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        planner = TaskPlanner(
            TaskPlannerModelClient(
                http_client=http_client,
                api_key="test-key",
                base_url="http://planner.test/v1",
                model="test-planner",
            )
        )
        request = self._request("帮我看看这个规则")
        plan, trace = planner.plan(request, None)

        self.assertEqual(plan.source, "classifier")
        self.assertEqual(plan.execution_route, "rag")
        self.assertTrue(trace.model_consulted)
        self.assertEqual(len(captured), 1)
        body = captured[0].content.decode("utf-8")
        self.assertNotIn(request.runtime_user_id, body)
        self.assertNotIn("currentPage", body)
        self.assertIn("候选工具", body)

    def test_agent_uses_route_plan_to_limit_model_visible_tools(self) -> None:
        tool_service = CapturingToolService()
        response = CustomerServiceAgent(tool_calling_service=tool_service).chat(
            self._request(
                "查询订单 SO20260420103000001-a1000001 的状态"
            )
        )

        self.assertEqual(tool_service.allowed_tool_names, ["get_order_status"])
        self.assertEqual(response.route_plan.execution_route, "tool")
        self.assertEqual(
            response.planner_trace.constrained_required_tools,
            ["get_order_status"],
        )
        self.assertEqual(
            response.session_state["route_plan"],
            response.route_plan.model_dump(),
        )
        self.assertEqual(response.session_state["agent_version"], "0.19.0")

    def test_agent_workflow_signal_does_not_claim_workflow_execution(self) -> None:
        response = CustomerServiceAgent().chat(
            self._request("请直接退款并赔付")
        )

        self.assertEqual(response.route_plan.execution_route, "workflow")
        self.assertTrue(response.route_plan.requires_workflow)
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.next_action, "transfer_to_human")
        self.assertFalse(
            response.session_state["risk_boundary"]["workflow_started"]
        )


if __name__ == "__main__":
    unittest.main()
