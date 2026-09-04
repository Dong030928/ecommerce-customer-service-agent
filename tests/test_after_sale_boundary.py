"""Offline regressions for the refund and received-return boundaries."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import ChatRequest  # noqa: E402
from embeddings.client import EmbeddingClient  # noqa: E402
from policies.after_sale_policy import AfterSalePolicyService  # noqa: E402
from rag.index_cache import reset_index_and_cache  # noqa: E402
from tools.tool_runtime import ToolRuntime  # noqa: E402


ORDER_ID = "SO20260420103000001-a1000001"


class AfterSaleEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        super().__init__(
            api_key="test-key",
            base_url="https://after-sale.example.invalid/v1",
            model="after-sale-test-embedding",
        )

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [
            1.0 if any(term in text for term in ["退款", "退货", "售后", "签收"]) else 0.0,
            1.0 if any(term in text for term in ["订单", "取消", "发货"]) else 0.0,
            1.0 if any(term in text for term in ["赔偿", "赔付", "补偿", "投诉"]) else 0.0,
        ]

    def embed(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class BoundaryEcommerceClient:
    def __init__(
        self,
        *,
        order_status: str = "PENDING_SHIPMENT",
        payment_status: str = "PAID",
        logistics_status: str = "NOT_SHIPPED",
        fulfillment_status: str | None = None,
        delivered_days_ago: int | None = None,
        returnable: bool | None = None,
    ) -> None:
        self.order_status = order_status
        self.payment_status = payment_status
        self.logistics_status = logistics_status
        self.fulfillment_status = fulfillment_status
        self.delivered_days_ago = delivered_days_ago
        self.returnable = returnable
        self.calls: list[tuple[str, str, str]] = []

    def get_order(self, order_id: str, runtime_user_id: str) -> dict:
        self.calls.append(("order", order_id, runtime_user_id))
        payload = {
            "orderNo": order_id,
            "status": self.order_status,
            "paymentStatus": self.payment_status,
            "userId": "PRIVATE-USER-ID",
            "remark": "PRIVATE-ORDER-REMARK",
        }
        if self.fulfillment_status is not None:
            payload["fulfillmentStatus"] = self.fulfillment_status
        if self.delivered_days_ago is not None:
            payload["deliveredAt"] = (
                date.today() - timedelta(days=self.delivered_days_ago)
            ).isoformat()
        if self.returnable is not None:
            payload["returnable"] = self.returnable
        return payload

    def get_logistics(self, order_id: str, runtime_user_id: str) -> dict:
        self.calls.append(("logistics", order_id, runtime_user_id))
        payload = {
            "status": self.logistics_status,
            "trackingNo": "PRIVATE-TRACKING-NUMBER",
        }
        if self.delivered_days_ago is not None:
            payload["deliveredAt"] = (
                date.today() - timedelta(days=self.delivered_days_ago)
            ).isoformat()
        return payload


class AfterSaleBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_index_and_cache()

    def tearDown(self) -> None:
        reset_index_and_cache()

    @staticmethod
    def request(message: str) -> ChatRequest:
        return ChatRequest(
            session_id="after-sale-boundary-test",
            runtime_user_id="TRUSTED-RUNTIME-USER",
            user_message=message,
        )

    def agent(self, client: BoundaryEcommerceClient) -> CustomerServiceAgent:
        return CustomerServiceAgent(
            embedding_client=AfterSaleEmbeddingClient(),
            answer_api_key="",
            after_sale_policy_service=AfterSalePolicyService(ToolRuntime(client)),
        )

    def test_missing_order_requests_clarification_without_business_read(self) -> None:
        client = BoundaryEcommerceClient()
        response = self.agent(client).chat(self.request("请马上帮我退款"))

        self.assertEqual(response.next_action, "ask_clarification")
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.citations, [])
        self.assertEqual(client.calls, [])
        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "needs_clarification",
        )

    def test_unshipped_paid_order_is_only_eligible_for_refund_application(self) -> None:
        client = BoundaryEcommerceClient()
        response = self.agent(client).chat(
            self.request(f"订单 {ORDER_ID} 直接退款，马上退钱")
        )

        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "eligible_for_application",
        )
        self.assertEqual(
            [record.action.tool_name for record in response.tool_calls],
            ["get_order_status", "get_order_logistics"],
        )
        self.assertTrue(response.citations)
        self.assertTrue(response.after_sale_assessment.policy_basis)
        self.assertIn("create_refund", response.after_sale_assessment.blocked_write_actions)
        self.assertFalse(response.session_state["risk_boundary"]["write_executed"])
        self.assertTrue(response.session_state["risk_boundary"]["workflow_started"])
        self.assertTrue(response.workflow.used_langgraph)
        self.assertEqual(response.next_action, "transfer_to_human")
        self.assertEqual(
            response.mcp_context.selected_tools,
            ["get_order_logistics", "get_order_status"],
        )

    def test_shipped_order_is_not_eligible_for_direct_refund(self) -> None:
        response = self.agent(
            BoundaryEcommerceClient(
                order_status="SHIPPED",
                logistics_status="IN_TRANSIT",
            )
        ).chat(self.request(f"订单 {ORDER_ID} 直接退款"))

        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "not_eligible",
        )
        self.assertIn("不能直接", response.answer)

    def test_received_return_checks_window_returnability_and_reason(self) -> None:
        response = self.agent(
            BoundaryEcommerceClient(
                order_status="DELIVERED",
                logistics_status="SIGNED",
                fulfillment_status="DELIVERED",
                delivered_days_ago=3,
                returnable=True,
            )
        ).chat(self.request(f"订单 {ORDER_ID} 已签收，七天无理由退货"))

        self.assertEqual(response.workflow.workflow_type, "received_return")
        self.assertEqual(response.workflow.pending_action, "prepare_return_application")
        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "eligible_for_application",
        )
        self.assertIn("签收时间", response.after_sale_assessment.evidence_checklist)
        self.assertFalse(response.session_state["risk_boundary"]["write_executed"])

    def test_received_return_outside_seven_day_window_is_not_eligible(self) -> None:
        response = self.agent(
            BoundaryEcommerceClient(
                order_status="DELIVERED",
                logistics_status="SIGNED",
                fulfillment_status="DELIVERED",
                delivered_days_ago=8,
                returnable=True,
            )
        ).chat(self.request(f"订单 {ORDER_ID} 已签收，七天无理由退货"))

        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "not_eligible",
        )
        self.assertIn("超过 7 天", response.after_sale_assessment.reasons[0])

    def test_received_return_requires_an_explicit_reason(self) -> None:
        response = self.agent(
            BoundaryEcommerceClient(
                order_status="DELIVERED",
                logistics_status="SIGNED",
                fulfillment_status="DELIVERED",
                delivered_days_ago=2,
                returnable=True,
            )
        ).chat(self.request(f"订单 {ORDER_ID} 我要退货"))

        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "needs_clarification",
        )
        self.assertEqual(response.next_action, "ask_clarification")
        self.assertIn("退货原因", response.answer)

    def test_compensation_always_requires_manual_review(self) -> None:
        response = self.agent(BoundaryEcommerceClient()).chat(
            self.request(f"订单 {ORDER_ID} 马上赔付我")
        )

        self.assertEqual(response.after_sale_assessment.action_type, "compensation")
        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "manual_review_required",
        )
        self.assertTrue(response.needs_human_approval)

    def test_raw_identity_and_business_payload_never_leave_observations(self) -> None:
        response = self.agent(BoundaryEcommerceClient()).chat(
            self.request(f"订单 {ORDER_ID} 直接退款")
        )
        serialized = response.model_dump_json()

        self.assertNotIn("PRIVATE-USER-ID", serialized)
        self.assertNotIn("PRIVATE-ORDER-REMARK", serialized)
        self.assertNotIn("PRIVATE-TRACKING-NUMBER", serialized)
        self.assertIn("order.userId", serialized)
        self.assertIn("logistics.trackingNo", serialized)


if __name__ == "__main__":
    unittest.main()
