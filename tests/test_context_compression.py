"""Regression tests for lesson-35 context compression and Sliding Window."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import (  # noqa: E402
    ChatRequest,
    ContextBuildReport,
    ContextItem,
    HistoryMessage,
)
from context.compression import ContextCompressor  # noqa: E402


ORDER_ID = "SO20260602103000009-a1000009"


class ContextCompressionTests(unittest.TestCase):
    @staticmethod
    def context_report() -> ContextBuildReport:
        return ContextBuildReport(
            selected_items=[
                ContextItem(
                    item_id="runtime-context-model-view",
                    source_type="runtime_context",
                    trust_level="trusted",
                    content="可信 Runtime Context：silver 会员。",
                    decision="可信当前事实。",
                ),
                ContextItem(
                    item_id="workflow-state",
                    source_type="workflow_state",
                    trust_level="verified",
                    content="退款工作流仍在等待人工审批。",
                    decision="流程边界。",
                ),
                ContextItem(
                    item_id="tool-observation-1",
                    source_type="tool_observation",
                    trust_level="verified",
                    content=f"订单 {ORDER_ID} 状态为 SHIPPED。",
                    decision="实时工具事实。",
                ),
                ContextItem(
                    item_id="rag-C1",
                    source_type="rag_snippet",
                    trust_level="external",
                    content="退款政策必须结合订单状态判断。",
                    decision="有引用的政策知识。",
                ),
                ContextItem(
                    item_id="user-message",
                    source_type="user_message",
                    trust_level="untrusted",
                    content=f"继续看 {ORDER_ID}",
                    decision="当前用户诉求。",
                ),
            ]
        )

    @staticmethod
    def history() -> list[HistoryMessage]:
        return [
            HistoryMessage(role="user", content=f"旧闲聊 {index}，与当前问题无关。")
            for index in range(8)
        ]

    def test_keeps_protected_recent_and_middle_relevant_items(self) -> None:
        history = self.history()
        history[1] = HistoryMessage(
            role="assistant",
            content=f"中间历史曾核对订单 {ORDER_ID}。",
        )
        report = ContextCompressor(max_context_tokens=40).compress(
            self.context_report(),
            history_messages=history,
            current_message=f"继续看 {ORDER_ID}",
        )

        kept_ids = {item.item_id for item in report.kept_items}
        self.assertTrue(
            {
                "runtime-context-model-view",
                "workflow-state",
                "tool-observation-1",
                "user-message",
                "rag-C1",
                "history-1",
                "history-4",
                "history-5",
                "history-6",
                "history-7",
            }.issubset(kept_ids)
        )
        self.assertTrue(report.dropped_items)
        self.assertIn("压缩摘要", report.compressed_summary)
        self.assertNotIn("旧闲聊 0", "\n".join(report.model_context))
        self.assertGreaterEqual(
            report.token_estimate_before,
            report.token_estimate_after,
        )

    def test_agent_exposes_compression_report_and_summary(self) -> None:
        request = ChatRequest(
            session_id="context-compression-test",
            runtime_user_id="U1002",
            runtime_member_level="silver",
            runtime_risk_level="low",
            user_message="你好",
            history_messages=self.history(),
        )

        response = CustomerServiceAgent().chat(request)

        self.assertTrue(response.compression_report.kept_items)
        self.assertEqual(response.compression_report.recent_window_size, 4)
        self.assertEqual(
            response.session_state["compression"]["dropped_count"],
            len(response.compression_report.dropped_items),
        )
        self.assertEqual(response.session_state["agent_version"], "0.33.0")


if __name__ == "__main__":
    unittest.main()
