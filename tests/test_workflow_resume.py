"""Native HITL decision, failure, concurrency and execution-isolation tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from api.schemas import ChatRequest, ChatResumeRequest
from integrations.ecommerce_client import EcommerceClientError
import test_langgraph_workflow as fixtures

ORDER_ID = fixtures.ORDER_ID


def approval_request(pending, decision="approved", **overrides):
    payload = dict(
        session_id=pending.session_id, workflow_id=pending.workflow.workflow_id,
        resume_token=pending.workflow.resume_token, reviewer_id="manager-01",
        reviewer_role="after_sale_manager", decision=decision,
    )
    payload.update(overrides)
    return ChatResumeRequest(**payload)


class WorkflowResumeTests(unittest.TestCase):
    def setUp(self):
        fixtures.reset_index_and_cache()
        self.addCleanup(fixtures.reset_index_and_cache)
        self.client = fixtures.WorkflowEcommerceClient()
        self.agent = fixtures.LangGraphWorkflowTests.agent(self.client)
        self.addCleanup(self.agent.close)
        self.pending = self.agent.chat(fixtures.LangGraphWorkflowTests.request(f"订单 {ORDER_ID} 直接退款"))
        self.workflow = self.agent._after_sale_workflow
        self.store = self.workflow.persistence.approvals

    def test_real_interrupt_and_primitive_state(self):
        snapshot = self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id))
        self.assertTrue(snapshot.interrupts)
        self.assertEqual(snapshot.next, ("human_review",))
        self.assertEqual(self.store.submission_count(), 0)
        self.assertNotIn("hooks", snapshot.values)
        self.assertIsInstance(snapshot.values["request"], dict)
        self.assertIsInstance(snapshot.values["assessment"], dict)
        self.assertEqual(snapshot.values["schema_version"], 1)
        json.dumps(snapshot.values)  # No live hooks, clients or custom Python objects.

    def test_approved_resumes_post_interrupt_nodes_without_rerunning_prefix(self):
        before = len(self.client.calls)
        result = self.agent.resume(approval_request(self.pending))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.workflow.node_history[-2:], ["recheck_business", "submit_application"])
        # Order recheck + logistics tool's order ownership read + logistics read.
        self.assertEqual(self.client.calls[before:], ["get_order", "get_order", "get_logistics"])
        snapshot = self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id))
        self.assertFalse(snapshot.interrupts)
        self.assertEqual(snapshot.next, ())
        self.assertEqual(self.store.submission_count(), 1)

    def test_rejection_and_conflicting_approval(self):
        before = len(self.client.calls)
        result = self.agent.resume(approval_request(self.pending, "rejected"))
        self.assertEqual(result.status, "rejected")
        replay = self.agent.resume(approval_request(self.pending, "rejected"))
        self.assertTrue(replay.resume_result.idempotent_replay)
        conflict = self.agent.resume(approval_request(self.pending))
        self.assertEqual(conflict.status, "blocked")
        self.assertFalse(conflict.resume_result.accepted)
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.store.submission_count(), 0)

    def test_approval_then_rejection_cannot_change_terminal_outcome(self):
        approved = self.agent.resume(approval_request(self.pending))
        conflict = self.agent.resume(approval_request(self.pending, "rejected"))
        self.assertEqual(conflict.status, "blocked")
        replay = self.agent.resume(approval_request(self.pending))
        self.assertEqual(replay.resume_result.request_id, approved.resume_result.request_id)
        self.assertEqual(self.store.submission_count(), 1)

    def test_needs_more_info_reinterrupts_then_can_approve(self):
        request = approval_request(self.pending, "needs_more_info", reviewer_note="补充退货原因")
        before = len(self.client.calls)
        result = self.agent.resume(request)
        self.assertEqual(result.status, "paused")
        self.assertEqual(result.approval.status, "needs_more_info")
        self.assertEqual(result.workflow.current_node, "human_review")
        snapshot = self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id))
        self.assertTrue(snapshot.interrupts)
        history = list(snapshot.values["node_history"])
        replay = self.agent.resume(request)
        self.assertTrue(replay.resume_result.idempotent_replay)
        self.assertEqual(self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id)).values["node_history"], history)
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.store.submission_count(), 0)
        self.assertEqual(self.agent.resume(approval_request(self.pending)).status, "completed")

    def test_concurrent_repeated_approval_commits_one_application(self):
        request = approval_request(self.pending)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.agent.resume(request), range(12)))
        self.assertTrue(all(result.status == "completed" for result in results))
        self.assertEqual(len({result.resume_result.request_id for result in results}), 1)
        self.assertEqual(sum(not result.resume_result.idempotent_replay for result in results), 1)
        self.assertEqual(self.store.submission_count(), 1)
        self.assertEqual(self.workflow.persistence._locks, {})

    def test_fresh_application_same_session_order_gets_distinct_thread(self):
        another = self.agent.chat(fixtures.LangGraphWorkflowTests.request(f"订单 {ORDER_ID} 直接退款"))
        self.assertNotEqual(another.workflow.workflow_id, self.pending.workflow.workflow_id)
        self.assertNotEqual(another.workflow.resume_token, self.pending.workflow.resume_token)
        bad = self.agent.resume(approval_request(another, resume_token=self.pending.workflow.resume_token))
        self.assertEqual(bad.status, "blocked")
        self.assertEqual(self.store.submission_count(), 0)

    def test_wrong_session_unknown_workflow_empty_reviewer_do_not_read_or_resume(self):
        before = len(self.client.calls)
        for overrides in ({"session_id": "other-session"}, {"workflow_id": "unknown"}, {"reviewer_id": "  "}):
            with self.subTest(overrides=overrides):
                response = self.agent.resume(approval_request(self.pending, **overrides))
                self.assertFalse(response.resume_result.accepted)
                self.assertIsNone(response.workflow)
        self.assertEqual(len(self.client.calls), before)
        self.assertTrue(self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id)).interrupts)

    def test_recheck_failure_blocks_and_never_uses_old_snapshot(self):
        self.client.fail_order = True
        result = self.agent.resume(approval_request(self.pending))
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.business_recheck["passed"])
        self.assertEqual(self.store.submission_count(), 0)

    def test_logistics_timeout_blocks_submission(self):
        with patch.object(self.client, "get_logistics", side_effect=EcommerceClientError("business_timeout", "超时")):
            result = self.agent.resume(approval_request(self.pending))
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.business_recheck["passed"])
        self.assertEqual(self.store.submission_count(), 0)

    def test_committed_business_result_survives_failed_graph_node(self):
        original = self.store.submit
        def fail_after_commit(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected-after-commit")
        with patch.object(self.store, "submit", side_effect=fail_after_commit):
            with self.assertRaises(RuntimeError):
                self.agent.resume(approval_request(self.pending))
        self.assertEqual(self.store.submission_count(), 1)
        before = len(self.client.calls)
        retry = self.agent.resume(approval_request(self.pending))
        self.assertEqual(retry.status, "completed")
        self.assertTrue(retry.resume_result.idempotent_replay)
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(self.store.submission_count(), 1)

    def test_uncommitted_decision_failure_retries_persisted_graph_node(self):
        request = approval_request(self.pending, "needs_more_info")
        with patch.object(self.store, "finish", side_effect=RuntimeError("injected-before-commit")):
            with self.assertRaises(RuntimeError):
                self.agent.resume(request)
        wrong_retry = self.agent.resume(approval_request(self.pending))
        self.assertEqual(wrong_retry.status, "blocked")
        retry = self.agent.resume(request)
        self.assertEqual(retry.status, "paused")
        self.assertTrue(self.workflow.graph.get_state(self.workflow._config(self.pending.workflow.workflow_id)).interrupts)

    def test_submission_and_terminal_outcome_are_one_transaction(self):
        with patch.object(self.store, "_write_result", side_effect=RuntimeError("injected-rollback")):
            with self.assertRaises(RuntimeError):
                self.agent.resume(approval_request(self.pending))
        self.assertEqual(self.store.submission_count(), 0)
        retry = self.agent.resume(approval_request(self.pending))
        self.assertEqual(retry.status, "completed")
        self.assertEqual(self.store.submission_count(), 1)

    def test_schema_mismatch_and_missing_native_snapshot_block(self):
        config = self.workflow._config(self.pending.workflow.workflow_id)
        self.workflow.graph.update_state(config, {"schema_version": 99})
        self.assertFalse(self.agent.resume(approval_request(self.pending)).resume_result.accepted)
        self.assertEqual(self.store.submission_count(), 0)

    def test_invalid_credential_does_not_leak_real_resume_token(self):
        bad = self.agent.resume(approval_request(self.pending, resume_token="wrong"))
        self.assertNotIn(self.pending.workflow.resume_token, bad.model_dump_json())

    def test_resume_never_replans_or_calls_answer_model(self):
        with patch.object(self.agent._task_planner, "plan", side_effect=AssertionError("no replanning")), \
             patch("agents.customer_service_agent.call_chat_model", side_effect=AssertionError("no answer LLM")):
            self.assertEqual(self.agent.resume(approval_request(self.pending)).status, "completed")

    def test_independent_users_and_threads_do_not_share_request_context(self):
        requests = [ChatRequest(
            session_id=f"isolated-{i}", runtime_user_id=f"user-{i}",
            user_message=f"订单 {ORDER_ID} 直接退款",
        ) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.agent.chat, requests))
        self.assertEqual(len({result.workflow.workflow_id for result in results}), 4)
        for request, result in zip(requests, results):
            record = self.store.get(request.session_id, result.workflow.workflow_id)
            self.assertEqual(record.requester_id, request.runtime_user_id)
            self.assertNotIn(request.runtime_user_id, result.workflow.model_dump_json())

    def test_all_frozen_order_fields_are_rechecked(self):
        mutations = {
            "order_status": ("status", "SHIPPED"),
            "payment_status": ("paymentStatus", "REFUNDED"),
            "total_amount": ("totalAmount", 199),
            "fulfillment_status": ("fulfillmentStatus", "SHIPPED"),
            "delivered_at": ("deliveredAt", "2026-10-05"),
            "returnable": ("returnable", True),
        }
        for field, (key, value) in mutations.items():
            with self.subTest(field=field):
                pending = self.agent.chat(fixtures.LangGraphWorkflowTests.request(f"订单 {ORDER_ID} 直接退款"))
                original = self.client.get_order
                def changed(order_id, user_id):
                    return {**original(order_id, user_id), key: value}
                with patch.object(self.client, "get_order", side_effect=changed):
                    result = self.agent.resume(approval_request(pending))
                self.assertEqual(result.status, "blocked")
                self.assertIn(field, result.business_recheck["mismatches"])
        self.assertEqual(self.store.submission_count(), 0)


if __name__ == "__main__":
    unittest.main()
