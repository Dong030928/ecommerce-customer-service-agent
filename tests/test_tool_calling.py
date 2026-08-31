"""Offline tests for realtime business facts and controlled Tool Calling."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import classify_intent  # noqa: E402
from api.schemas import (  # noqa: E402
    ChatRequest,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from integrations.ecommerce_client import EcommerceClient  # noqa: E402
from tools.contracts import TOOL_SPECS  # noqa: E402
from tools.tool_calling import ToolCallingOutcome  # noqa: E402
from tools.tool_runtime import ToolRuntime, validate_tool_action  # noqa: E402


class SuccessfulToolService:
    """Deterministic service double for the Agent-level realtime route."""

    def run(
        self,
        request: ChatRequest,
        intent: str,
        clarification_plan: object | None = None,
    ) -> ToolCallingOutcome:
        del request, intent, clarification_plan
        record = ToolCallRecord(
            action=ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": "SO20260420103000001-a1000001"},
                reason="查询实时订单状态。",
            ),
            observation=ToolObservation(
                tool_name="get_order_status",
                status="success",
                summary="订单当前状态为 SHIPPED。",
                data={"order_id": "SO20260420103000001-a1000001", "status": "SHIPPED"},
            ),
        )
        return ToolCallingOutcome(
            answer="我通过实时业务工具查到：订单当前状态为 SHIPPED。",
            tool_calls=[record],
            state={
                "create_agent": True,
                "answer_source": "sanitized_tool_observation",
                "model_final_wording_used": False,
            },
            used_model=True,
            model_name="test-tool-model",
        )


class ToolCallingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            path = request.url.path
            if path.endswith("/logistics"):
                data = {
                    "company": "顺丰",
                    "trackingNo": "SF10001",
                    "status": "IN_TRANSIT",
                    "latestUpdate": "已到达上海转运中心",
                }
            elif path.startswith("/api/orders/"):
                data = {
                    "orderNo": "SO20260420103000001-a1000001",
                    "userId": "PRIVATE-USER-ID",
                    "customerName": "PRIVATE-CUSTOMER-NAME",
                    "status": "SHIPPED",
                    "paymentStatus": "PAID",
                    "totalAmount": "299.00",
                }
            elif path.startswith("/api/refund/requests/"):
                data = {
                    "requestId": "RF-1001",
                    "orderNo": "SO20260420103000001-a1000001",
                    "userId": "PRIVATE-USER-ID",
                    "amount": "299.00",
                    "status": "reviewing",
                    "approvalId": "AP-1001",
                }
            elif path == "/api/products":
                data = [
                    {
                        "id": 1,
                        "code": "SKU-AUD-101",
                        "name": "降噪蓝牙耳机",
                        "price": "299.00",
                        "stock": 18,
                        "active": True,
                    }
                ]
            else:
                return httpx.Response(404, json={"success": False})
            return httpx.Response(200, json={"success": True, "data": data})

        self.http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.ecommerce_client = EcommerceClient(
            base_url="http://ecommerce.test",
            service_token="service-token-secret",
            http_client=self.http_client,
        )
        self.runtime = ToolRuntime(self.ecommerce_client)
        self.request = ChatRequest(
            session_id="tool-test",
            runtime_user_id="PRIVATE-USER-ID",
            runtime_nickname="PRIVATE-NICKNAME",
            user_message=(
                "请查 SO20260420103000001-a1000001 的物流"
            ),
        )

    def tearDown(self) -> None:
        self.http_client.close()

    def test_order_logistics_uses_delegated_identity_and_sanitizes_result(self) -> None:
        observation = self.runtime.execute(
            ToolAction(
                tool_name="get_order_logistics",
                arguments={"order_id": "SO20260420103000001-a1000001"},
                reason="test",
            ),
            self.request,
        )

        self.assertEqual(observation.status, "success")
        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            self.assertEqual(
                request.headers["X-Agent-User-Id"],
                "PRIVATE-USER-ID",
            )
            self.assertEqual(
                request.headers["X-Agent-Service-Token"],
                "service-token-secret",
            )
        serialized = observation.model_dump_json()
        self.assertNotIn("PRIVATE-USER-ID", serialized)
        self.assertNotIn("PRIVATE-CUSTOMER-NAME", serialized)
        self.assertIn("IN_TRANSIT", serialized)

    def test_model_cannot_supply_or_override_runtime_identity(self) -> None:
        observation = validate_tool_action(
            ToolAction(
                tool_name="get_order_status",
                arguments={
                    "order_id": "SO20260420103000001-a1000001",
                    "user_id": "ATTACKER",
                },
                reason="test",
            )
        )

        self.assertIsNotNone(observation)
        self.assertEqual(observation.error_code, "tool_arguments_not_allowed")
        self.assertEqual(self.requests, [])
        for spec in TOOL_SPECS.values():
            self.assertNotIn("user_id", spec.parameters_schema)

    def test_refund_status_reads_real_refund_endpoint_and_removes_user_id(self) -> None:
        observation = self.runtime.execute(
            ToolAction(
                tool_name="get_refund_status",
                arguments={"refund_request_id": "RF-1001"},
                reason="test",
            ),
            self.request,
        )

        self.assertEqual(observation.status, "success")
        self.assertEqual(self.requests[0].url.path, "/api/refund/requests/RF-1001")
        self.assertNotIn("PRIVATE-USER-ID", observation.model_dump_json())
        self.assertIn("reviewing", observation.summary)

    def test_product_inventory_uses_live_catalog(self) -> None:
        observation = self.runtime.execute(
            ToolAction(
                tool_name="get_product_inventory",
                arguments={"sku": "SKU-AUD-101"},
                reason="test",
            ),
            self.request,
        )

        self.assertEqual(observation.status, "success")
        self.assertIn("299.00", observation.summary)
        self.assertIn("18", observation.summary)
        self.assertEqual(self.requests[0].url.params["keyword"], "SKU-AUD-101")

    def test_refund_status_intent_has_priority_over_refund_request(self) -> None:
        result = classify_intent("帮我查一下退款进度，申请号 RF-1001")

        self.assertEqual(result.intent, "refund_status_query")
        self.assertEqual(result.source, "rules")

    def test_explicit_business_ids_route_to_realtime_intents(self) -> None:
        refund = classify_intent("RF-1001 现在是什么状态")
        product = classify_intent("SKU-AUD-101 现在还有货吗")

        self.assertEqual(refund.intent, "refund_status_query")
        self.assertEqual(product.intent, "product_consult")
        self.assertEqual(refund.confidence, 0.95)
        self.assertEqual(product.confidence, 0.95)

    def test_agent_returns_tool_record_and_skips_rag_for_realtime_fact(self) -> None:
        from agents.customer_service_agent import CustomerServiceAgent

        agent = CustomerServiceAgent(tool_calling_service=SuccessfulToolService())
        response = agent.chat(
            ChatRequest(
                session_id="agent-tool-test",
                runtime_user_id="PRIVATE-USER-ID",
                user_message="查询订单 SO20260420103000001-a1000001 的状态",
            )
        )

        self.assertEqual(response.intent, "order_query")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(
            response.tool_calls[0].action.tool_name,
            "get_order_status",
        )
        self.assertEqual(
            response.session_state["rag"]["status"],
            "skipped_realtime_tool_route",
        )
        self.assertEqual(response.citations, [])
        self.assertEqual(response.next_action, "answer_user")
        self.assertFalse(
            response.session_state["tool_calling"]["raw_tool_result_exposed"]
        )
        self.assertNotIn("PRIVATE-USER-ID", response.answer)


if __name__ == "__main__":
    unittest.main()
