"""Offline regression tests for lesson 21 error and degradation policies."""

from __future__ import annotations

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
from integrations.ecommerce_client import (  # noqa: E402
    EcommerceClient,
    EcommerceClientError,
)
from degradation.fallbacks import is_high_risk_write_request  # noqa: E402
from tools.tool_calling import ToolCallingOutcome  # noqa: E402
from tools.tool_runtime import ToolRuntime  # noqa: E402


ORDER_ID = "SO20260420103000001-a1000001"


def request(message: str = "查询订单状态") -> ChatRequest:
    return ChatRequest(
        session_id="degradation-test",
        runtime_user_id="trusted-user",
        user_message=message,
    )


class FlakyOrderClient:
    def __init__(self, failures: int, code: str = "business_timeout") -> None:
        self.failures = failures
        self.code = code
        self.calls = 0

    def get_order(self, order_id: str, runtime_user_id: str) -> dict:
        del runtime_user_id
        self.calls += 1
        if self.calls <= self.failures:
            raise EcommerceClientError(self.code, "安全错误信息")
        return {
            "orderNo": order_id,
            "status": "SHIPPED",
            "paymentStatus": "PAID",
        }


class FailedToolService:
    def run(self, *args: object, **kwargs: object) -> ToolCallingOutcome:
        del args, kwargs
        return ToolCallingOutcome(
            answer="不应直接使用这个回答",
            tool_calls=[],
            state={"answer_source": "safe_tool_fallback"},
            used_model=False,
            error="RuntimeError",
        )


class RefundStatusService:
    def run(self, *args: object, **kwargs: object) -> ToolCallingOutcome:
        del args, kwargs
        observation = ToolObservation(
            tool_name="get_refund_status",
            status="success",
            summary="退款申请 RF-1001 当前状态为 reviewing。",
        )
        return ToolCallingOutcome(
            answer=observation.summary,
            tool_calls=[
                ToolCallRecord(
                    action=ToolAction(
                        tool_name="get_refund_status",
                        arguments={"refund_request_id": "RF-1001"},
                        reason="查询退款进度。",
                    ),
                    observation=observation,
                )
            ],
            state={"answer_source": "compressed_observation_fallback"},
            used_model=True,
        )


class DegradationTests(unittest.TestCase):
    def test_http_timeout_has_a_distinct_safe_error_code(self) -> None:
        def handler(http_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("private network detail", request=http_request)

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = EcommerceClient(
            base_url="http://ecommerce.test",
            service_token="service-token",
            http_client=http_client,
        )
        try:
            with self.assertRaises(EcommerceClientError) as caught:
                client.get_order(ORDER_ID, "trusted-user")
            self.assertEqual(caught.exception.code, "business_timeout")
            self.assertNotIn("private network detail", caught.exception.safe_message)
        finally:
            http_client.close()

    def test_read_only_timeout_retries_once_then_succeeds(self) -> None:
        client = FlakyOrderClient(failures=1)
        observation = ToolRuntime(client).execute(
            ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": ORDER_ID},
                reason="test",
            ),
            request(),
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(observation.status, "success")
        self.assertEqual(observation.attempts, 2)
        self.assertEqual(observation.error_category, "none")

    def test_repeated_timeout_stops_after_two_attempts(self) -> None:
        client = FlakyOrderClient(failures=3)
        observation = ToolRuntime(client).execute(
            ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": ORDER_ID},
                reason="test",
            ),
            request(),
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(observation.status, "error")
        self.assertEqual(observation.attempts, 2)
        self.assertEqual(observation.error_category, "timeout")
        self.assertEqual(observation.next_action, "fallback_answer")

    def test_non_timeout_error_is_not_retried(self) -> None:
        client = FlakyOrderClient(failures=3, code="business_fact_not_found")
        observation = ToolRuntime(client).execute(
            ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": ORDER_ID},
                reason="test",
            ),
            request(),
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(observation.error_category, "not_found")
        self.assertEqual(observation.attempts, 1)

    def test_high_risk_write_only_allows_read_only_evidence_collection(self) -> None:
        response = CustomerServiceAgent().chat(
            request(f"请马上帮我取消订单 {ORDER_ID} 并退款")
        )

        self.assertTrue(response.degraded)
        self.assertEqual(response.risk_level, "high")
        self.assertTrue(response.needs_human_approval)
        self.assertEqual(response.next_action, "transfer_to_human")
        self.assertEqual(
            [record.action.tool_name for record in response.tool_calls],
            ["get_order_status", "get_order_logistics"],
        )
        self.assertTrue(
            all(
                record.action.tool_name
                not in response.after_sale_assessment.blocked_write_actions
                for record in response.tool_calls
            )
        )
        self.assertEqual(response.citations, [])
        self.assertEqual(
            response.after_sale_assessment.eligibility_status,
            "blocked",
        )
        self.assertFalse(response.session_state["risk_boundary"]["write_executed"])
        self.assertEqual(
            response.session_state["degradation"]["error_category"],
            "high_risk_write_blocked",
        )

    def test_refund_status_query_is_not_mistaken_for_a_write(self) -> None:
        response = CustomerServiceAgent(
            tool_calling_service=RefundStatusService()
        ).chat(request("帮我查退款进度，申请号 RF-1001"))

        self.assertFalse(response.needs_human_approval)
        self.assertEqual(response.risk_level, "medium")
        self.assertEqual(response.next_action, "answer_user")
        self.assertFalse(response.degraded)

    def test_refund_policy_question_and_negation_are_not_write_requests(self) -> None:
        self.assertFalse(is_high_risk_write_request("请问如何申请退款？"))
        self.assertFalse(is_high_risk_write_request("先不要帮我退款"))
        self.assertTrue(is_high_risk_write_request("请马上帮我退款"))

    def test_tool_planning_failure_uses_model_unavailable_fallback(self) -> None:
        response = CustomerServiceAgent(
            tool_calling_service=FailedToolService()
        ).chat(request(f"查询订单 {ORDER_ID} 的状态"))

        self.assertTrue(response.degraded)
        self.assertEqual(response.next_action, "fallback_answer")
        self.assertEqual(
            response.session_state["degradation"]["error_category"],
            "model_unavailable",
        )
        self.assertIn("模型当前不可用", response.answer)


if __name__ == "__main__":
    unittest.main()
