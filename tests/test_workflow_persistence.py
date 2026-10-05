"""SQLite graph snapshots and business outcomes survive actual process restart."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from state.native_checkpoint import WorkflowPersistence
from test_workflow_resume import approval_request
from workflow_restart_probe import build_agent
import test_langgraph_workflow as fixtures
import test_after_sale_boundary as boundary_fixtures


class WorkflowPersistenceTests(unittest.TestCase):
    def setUp(self):
        fixtures.reset_index_and_cache()
        self.addCleanup(fixtures.reset_index_and_cache)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = fixtures.WorkflowEcommerceClient()

    def agent(self):
        agent = build_agent(WorkflowPersistence.sqlite(
            self.root / "graph.sqlite3", self.root / "approval.sqlite3",
        ), self.client)
        self.addCleanup(agent.close)
        return agent

    def pending(self, agent):
        return agent.chat(fixtures.LangGraphWorkflowTests.request(f"订单 {fixtures.ORDER_ID} 直接退款"))

    def test_new_agent_restores_waiting_native_graph(self):
        first = self.agent()
        pending = self.pending(first)
        token = pending.workflow.resume_token
        first.close()
        second = self.agent()
        result = second.resume(approval_request(pending))
        self.assertEqual(result.status, "completed")
        self.assertIn("submit_application", result.workflow.node_history)
        self.assertEqual(second._after_sale_workflow.persistence.approvals.submission_count(), 1)
        self.assertNotIn(token, result.model_dump_json())

    def test_sqlite_concurrent_repeated_approval_commits_one_application(self):
        agent = self.agent()
        request = approval_request(self.pending(agent))
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: agent.resume(request), range(12)))
        self.assertTrue(all(result.status == "completed" for result in results))
        self.assertEqual(len({result.resume_result.request_id for result in results}), 1)
        self.assertEqual(sum(not result.resume_result.idempotent_replay for result in results), 1)
        self.assertEqual(agent._after_sale_workflow.persistence.approvals.submission_count(), 1)

    def test_received_return_resumes_after_sqlite_restart(self):
        self.client = boundary_fixtures.BoundaryEcommerceClient(
            order_status="DELIVERED", logistics_status="SIGNED",
            fulfillment_status="DELIVERED", delivered_days_ago=3, returnable=True,
        )
        first = self.agent()
        pending = first.chat(fixtures.LangGraphWorkflowTests.request(
            f"订单 {fixtures.ORDER_ID} 已签收，七天无理由退货",
        ))
        self.assertEqual(pending.workflow.workflow_type, "received_return")
        self.assertEqual(pending.workflow.status, "paused")
        first.close()
        second = self.agent()
        result = second.resume(approval_request(pending))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.workflow.workflow_type, "received_return")
        self.assertTrue(result.business_recheck["passed"])
        self.assertEqual(second._after_sale_workflow.persistence.approvals.submission_count(), 1)

    def test_completed_outcome_and_idempotency_survive_recreation(self):
        first = self.agent()
        pending = self.pending(first)
        completed = first.resume(approval_request(pending))
        first.close()
        second = self.agent()
        before = len(self.client.calls)
        replay = second.resume(approval_request(pending))
        self.assertTrue(replay.resume_result.idempotent_replay)
        self.assertEqual(replay.resume_result.request_id, completed.resume_result.request_id)
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(second._after_sale_workflow.persistence.approvals.submission_count(), 1)

    def test_rejected_terminal_decision_survives_restart(self):
        first = self.agent()
        pending = self.pending(first)
        self.assertEqual(first.resume(approval_request(pending, "rejected")).status, "rejected")
        first.close()
        second = self.agent()
        self.assertEqual(second.resume(approval_request(pending)).status, "blocked")
        replay = second.resume(approval_request(pending, "rejected"))
        self.assertEqual(replay.status, "rejected")
        self.assertTrue(replay.resume_result.idempotent_replay)

    def test_more_info_waiting_state_survives_restart(self):
        first = self.agent()
        pending = self.pending(first)
        request = approval_request(pending, "needs_more_info")
        first.resume(request)
        first.close()
        second = self.agent()
        self.assertTrue(second.resume(request).resume_result.idempotent_replay)
        self.assertEqual(second.resume(approval_request(pending)).status, "completed")

    def test_fault_after_business_commit_survives_restart_without_double_submission(self):
        first = self.agent()
        pending = self.pending(first)
        store = first._after_sale_workflow.persistence.approvals
        original = store.submit
        def fail_after_commit(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected-after-business-commit")
        with patch.object(store, "submit", side_effect=fail_after_commit):
            with self.assertRaises(RuntimeError):
                first.resume(approval_request(pending))
        first.close()
        second = self.agent()
        replay = second.resume(approval_request(pending))
        self.assertTrue(replay.resume_result.idempotent_replay)
        self.assertEqual(replay.status, "completed")
        self.assertEqual(second._after_sale_workflow.persistence.approvals.submission_count(), 1)

    def test_actual_child_process_exit_and_restart(self):
        probe = Path(__file__).with_name("workflow_restart_probe.py")
        env = {**os.environ, "AGENT_DISABLE_LLM": "1", "PYTHONIOENCODING": "utf-8"}
        seed = subprocess.run(
            [sys.executable, str(probe), "seed", str(self.root)],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=30, check=True,
        )
        request = json.loads(seed.stdout)
        resume = subprocess.run(
            [sys.executable, str(probe), "resume", str(self.root)],
            input=json.dumps(request), capture_output=True, text=True,
            encoding="utf-8", env=env, timeout=30, check=True,
        )
        result = json.loads(resume.stdout)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["business_recheck"]["passed"])
        repeat = subprocess.run(
            [sys.executable, str(probe), "resume", str(self.root)],
            input=json.dumps(request), capture_output=True, text=True,
            encoding="utf-8", env=env, timeout=30, check=True,
        )
        replay = json.loads(repeat.stdout)
        self.assertEqual(replay["resume_result"]["request_id"], result["resume_result"]["request_id"])
        self.assertTrue(replay["resume_result"]["idempotent_replay"])

    def test_business_store_contains_digest_not_raw_resume_token(self):
        agent = self.agent()
        pending = self.pending(agent)
        with sqlite3.connect(self.root / "approval.sqlite3") as connection:
            payload = connection.execute("SELECT snapshot_json FROM approvals").fetchone()[0]
        connection.close()
        self.assertNotIn(pending.workflow.resume_token, payload)
        self.assertIn("resume_token_digest", payload)

    def test_unsupported_store_schema_and_invalid_paths_fail_closed(self):
        agent = self.agent()
        agent.close()
        with sqlite3.connect(self.root / "approval.sqlite3") as connection:
            connection.execute("UPDATE workflow_store_meta SET version=99")
        connection.close()
        with self.assertRaises(RuntimeError):
            WorkflowPersistence.sqlite(self.root / "graph.sqlite3", self.root / "approval.sqlite3")
        with self.assertRaises(RuntimeError):
            WorkflowPersistence.sqlite(self.root, self.root / "new.sqlite3")
        with self.assertRaises(RuntimeError):
            WorkflowPersistence.sqlite(self.root / "same.sqlite3", self.root / "same.sqlite3")

    def test_missing_graph_snapshot_blocks_existing_business_approval(self):
        agent = self.agent()
        pending = self.pending(agent)
        agent._after_sale_workflow.persistence.checkpointer.delete_thread(pending.workflow.workflow_id)
        result = agent.resume(approval_request(pending))
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.resume_result.accepted)
        self.assertEqual(agent._after_sale_workflow.persistence.approvals.submission_count(), 0)

    def test_closed_resource_never_silently_falls_back_to_memory(self):
        agent = self.agent()
        pending = self.pending(agent)
        agent.close()
        with self.assertRaises(RuntimeError):
            agent.resume(approval_request(pending))


if __name__ == "__main__":
    unittest.main()
