"""Offline tests for internal ToolResult and compressed Observation boundaries."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from api.schemas import (  # noqa: E402
    ChatRequest,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from tools.tool_calling import _final_model_wording  # noqa: E402
from tools.tool_runtime import ToolRuntime  # noqa: E402


class PrivatePayloadClient:
    def get_order(self, order_id: str, runtime_user_id: str) -> dict:
        return {
            "orderNo": order_id,
            "userId": runtime_user_id,
            "customerName": "PRIVATE-CUSTOMER-NAME",
            "status": "SHIPPED",
            "paymentStatus": "PAID",
            "totalAmount": "299.00",
            "items": [{"name": "PRIVATE-ITEM-DETAIL"}],
            "remark": "PRIVATE-ORDER-REMARK",
        }

    def get_logistics(self, order_id: str, runtime_user_id: str) -> dict:
        del order_id, runtime_user_id
        return {
            "company": "顺丰",
            "trackingNo": "PRIVATE-TRACKING-NO",
            "status": "IN_TRANSIT",
            "latestUpdate": "已到达上海转运中心",
            "estimatedDelivery": "2026-06-08",
            "exceptionReason": "PRIVATE-EXCEPTION",
            "events": [
                {
                    "occurredAt": "2026-06-02T15:00:00",
                    "content": "PRIVATE-FULL-LOGISTICS-EVENT",
                }
            ],
        }

    def list_products(self, query: str) -> list[dict]:
        return [
            {
                "code": query,
                "name": "降噪蓝牙耳机",
                "price": "299.00",
                "stock": 18,
                "active": True,
                "description": "PRIVATE-LONG-DESCRIPTION",
                "highlights": ["PRIVATE-HIGHLIGHT"],
                "promotion": {"rule": "PRIVATE-PROMOTION-RULE"},
            }
        ]

    def get_refund_status(
        self,
        refund_request_id: str,
        runtime_user_id: str,
    ) -> dict:
        return {
            "requestId": refund_request_id,
            "orderNo": "SO20260420103000001-a1000001",
            "userId": runtime_user_id,
            "status": "reviewing",
            "amount": "299.00",
            "reason": "PRIVATE-REFUND-REASON",
            "approvalId": "PRIVATE-APPROVAL-ID",
        }


class ToolObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ToolRuntime(PrivatePayloadClient())
        self.request = ChatRequest(
            session_id="observation-test",
            runtime_user_id="PRIVATE-USER-ID",
            user_message="查询物流",
        )

    def test_raw_logistics_result_is_compressed_before_public_boundary(self) -> None:
        action = ToolAction(
            tool_name="get_order_logistics",
            arguments={"order_id": "SO20260420103000001-a1000001"},
            reason="test",
        )
        raw = self.runtime.execute_raw(action, self.request)
        observation = self.runtime.execute(action, self.request)

        raw_json = raw.model_dump_json()
        public_json = observation.model_dump_json()
        self.assertIn("PRIVATE-USER-ID", raw_json)
        self.assertIn("PRIVATE-TRACKING-NO", raw_json)
        self.assertIn("PRIVATE-FULL-LOGISTICS-EVENT", raw_json)
        self.assertNotIn("PRIVATE-USER-ID", public_json)
        self.assertNotIn("PRIVATE-TRACKING-NO", public_json)
        self.assertNotIn("PRIVATE-FULL-LOGISTICS-EVENT", public_json)
        self.assertIn("logistics.trackingNo", observation.omitted_fields)
        self.assertIn("logistics.events", observation.omitted_fields)
        self.assertEqual(observation.next_action, "answer_user")
        self.assertEqual(observation.facts["logistics_status"], "IN_TRANSIT")

    def test_product_observation_omits_long_description_and_promotion(self) -> None:
        observation = self.runtime.execute(
            ToolAction(
                tool_name="get_product_inventory",
                arguments={"sku": "SKU-AUD-101"},
                reason="test",
            ),
            self.request,
        )

        serialized = observation.model_dump_json()
        self.assertNotIn("PRIVATE-LONG-DESCRIPTION", serialized)
        self.assertNotIn("PRIVATE-PROMOTION-RULE", serialized)
        self.assertIn("product.description", observation.omitted_fields)
        self.assertIn("product.promotion", observation.omitted_fields)
        self.assertEqual(observation.facts["inventory"], 18)

    def test_refund_observation_keeps_status_but_removes_private_reason(self) -> None:
        observation = self.runtime.execute(
            ToolAction(
                tool_name="get_refund_status",
                arguments={"refund_request_id": "RF-1001"},
                reason="test",
            ),
            self.request,
        )

        serialized = observation.model_dump_json()
        self.assertEqual(observation.facts["refund_status"], "reviewing")
        self.assertNotIn("PRIVATE-REFUND-REASON", serialized)
        self.assertNotIn("PRIVATE-APPROVAL-ID", serialized)
        self.assertIn("refund.reason", observation.omitted_fields)

    def test_final_model_wording_requires_answerable_observation(self) -> None:
        ai_message_type = type("AIMessage", (), {})
        message = ai_message_type()
        message.content = "基于安全 Observation 生成的最终回答。"
        message.tool_calls = []
        success = ToolCallRecord(
            action=ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": "SO20260420103000001-a1000001"},
                reason="test",
            ),
            observation=ToolObservation(
                tool_name="get_order_status",
                status="success",
                summary="订单已发货。",
                facts={"order_status": "SHIPPED"},
                next_action="answer_user",
            ),
        )
        blocked = success.model_copy(
            update={
                "observation": success.observation.model_copy(
                    update={"next_action": "ask_clarification"}
                )
            }
        )

        self.assertEqual(
            _final_model_wording([message], [success]),
            "基于安全 Observation 生成的最终回答。",
        )
        self.assertIsNone(_final_model_wording([message], [blocked]))


if __name__ == "__main__":
    unittest.main()
