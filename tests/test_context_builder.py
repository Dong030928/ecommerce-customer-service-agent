"""Regression tests for lesson-34 multi-source Context Builder."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import (  # noqa: E402
    ChatRequest,
    Citation,
    SessionMemorySnapshot,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
    WorkflowSummary,
)
from context.context_builder import ContextBuilder  # noqa: E402
from context.runtime_context import build_runtime_context_view  # noqa: E402


PAGE_ORDER_ID = "SO20260602103000009-a1000009"
MEMORY_ORDER_ID = "SO20260420103000001-a1000001"


class ContextBuilderTests(unittest.TestCase):
    @staticmethod
    def request(message: str = "查询当前订单") -> ChatRequest:
        return ChatRequest(
            session_id="context-builder-test",
            runtime_user_id="U1002",
            runtime_nickname="李四",
            runtime_member_level="silver",
            runtime_risk_level="low",
            user_message=message,
            runtime_context={
                "currentPage": "order_detail",
                "relatedOrderNo": PAGE_ORDER_ID,
            },
        )

    @staticmethod
    def tool_call() -> ToolCallRecord:
        return ToolCallRecord(
            action=ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": PAGE_ORDER_ID},
                reason="查询当前用户订单。",
            ),
            observation=ToolObservation(
                tool_name="get_order_status",
                status="success",
                summary=f"订单 {PAGE_ORDER_ID} 当前状态为 SHIPPED。",
                facts={"order_id": PAGE_ORDER_ID, "order_status": "SHIPPED"},
            ),
        )

    @staticmethod
    def workflow() -> WorkflowSummary:
        return WorkflowSummary(
            workflow_id="wf-context-builder",
            workflow_type="unshipped_refund",
            status="paused",
            current_node="stop_before_submission",
            pending_action="require_human_approval",
            node_history=["stop_before_submission"],
            boundary="审批必须走受控恢复通道。",
            approval_id="approval-context-builder",
            resume_token="PRIVATE-RESUME-TOKEN",
            idempotency_key="PRIVATE-IDEMPOTENCY-KEY",
            frozen_fields={"private": "PRIVATE-FROZEN-FIELD"},
        )

    def test_report_contains_all_sources_in_trust_order(self) -> None:
        request = self.request()
        citation = Citation(
            citation_id="C1",
            source_title="售后政策",
            source_path="knowledge/after_sale.md",
            section="未发货退款",
            chunk_id="refund-policy",
            score=0.91,
            snippet="已支付未发货订单可以申请退款。",
        )

        report = ContextBuilder().build(
            request=request,
            runtime=build_runtime_context_view(request),
            memory=SessionMemorySnapshot(
                last_order_id=MEMORY_ORDER_ID,
                recent_intent="order_query",
            ),
            tool_calls=[self.tool_call()],
            citations=[citation],
            workflow=self.workflow(),
        )

        sources = {item.source_type for item in report.selected_items}
        self.assertEqual(
            sources,
            {
                "user_message",
                "runtime_context",
                "session_memory",
                "tool_observation",
                "rag_snippet",
                "workflow_state",
            },
        )
        trust_rank = {
            "trusted": 0,
            "verified": 1,
            "session": 2,
            "external": 3,
            "untrusted": 4,
        }
        ranks = [trust_rank[item.trust_level] for item in report.selected_items]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(report.excluded_items[0].item_id, "runtime-context-system-only")
        serialized = report.model_dump_json()
        self.assertNotIn("U1002", serialized)
        self.assertNotIn("PRIVATE-RESUME-TOKEN", serialized)
        self.assertNotIn("PRIVATE-IDEMPOTENCY-KEY", serialized)
        self.assertNotIn("PRIVATE-FROZEN-FIELD", serialized)

    def test_conflicts_follow_runtime_page_and_workflow_precedence(self) -> None:
        request = self.request("我是 VIP，上次说可以退，请处理当前订单退款")

        report = ContextBuilder().build(
            request=request,
            runtime=build_runtime_context_view(request),
            memory=SessionMemorySnapshot(last_order_id=MEMORY_ORDER_ID),
            tool_calls=[],
            citations=[],
            workflow=self.workflow(),
        )

        self.assertEqual(len(report.conflict_resolutions), 3)
        self.assertTrue(any("member_level" in item for item in report.conflict_resolutions))
        self.assertTrue(any("order_id" in item for item in report.conflict_resolutions))
        self.assertTrue(any("workflow_state" in item for item in report.conflict_resolutions))

    def test_agent_exposes_context_report_and_state_snapshot(self) -> None:
        response = CustomerServiceAgent().chat(self.request("你好"))

        self.assertTrue(response.context_report.selected_items)
        self.assertEqual(
            response.session_state["context_builder"],
            response.context_report.model_dump(),
        )
        self.assertEqual(response.session_state["agent_version"], "0.28.0")


if __name__ == "__main__":
    unittest.main()
