"""HTTP compatibility and application-owned workflow persistence lifecycle."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient
from main import create_app
from state.native_checkpoint import WorkflowPersistence
from test_workflow_resume import approval_request
from workflow_restart_probe import build_agent
import test_langgraph_workflow as fixtures


class WorkflowApiTests(unittest.TestCase):
    def setUp(self):
        fixtures.reset_index_and_cache()
        self.addCleanup(fixtures.reset_index_and_cache)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def app(self):
        return create_app(
            persistence_factory=lambda: WorkflowPersistence.sqlite(
                self.root / "graph.sqlite3", self.root / "approval.sqlite3",
            ),
            agent_factory=build_agent,
        )

    def test_resume_http_contract_and_shutdown_restart(self):
        app = self.app()
        with TestClient(app) as client:
            response = client.post("/chat", json=fixtures.LangGraphWorkflowTests.request(
                f"订单 {fixtures.ORDER_ID} 直接退款",
            ).model_dump(mode="json"))
            self.assertEqual(response.status_code, 200)
            pending = response.json()
            request = dict(
                session_id=pending["session_id"], workflow_id=pending["workflow"]["workflow_id"],
                resume_token=pending["workflow"]["resume_token"], reviewer_id="manager-01",
                reviewer_role="after_sale_manager", decision="approved",
            )
            persistence = app.state.agent._after_sale_workflow.persistence
        self.assertTrue(persistence._closed)
        self.assertIsNone(app.state.agent)
        with TestClient(self.app()) as client:
            result = client.post("/chat/resume", json=request)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "completed")
            self.assertTrue(result.json()["business_recheck"]["passed"])
            self.assertFalse(result.json()["session_state"]["external_business_write_executed"])
            invalid = client.post("/chat/resume", json={**request, "decision": "free-text-approve"})
            self.assertEqual(invalid.status_code, 422)
            replay = client.post("/chat/resume", json=request).json()
            self.assertTrue(replay["resume_result"]["idempotent_replay"])

    def test_storage_error_is_safe_service_error(self):
        app = self.app()
        with TestClient(app) as client:
            agent = app.state.agent
            pending = agent.chat(fixtures.LangGraphWorkflowTests.request(f"订单 {fixtures.ORDER_ID} 直接退款"))
            with patch.object(agent._after_sale_workflow.persistence.approvals, "get",
                              side_effect=RuntimeError("售后存储不可用。")):
                response = client.post("/chat/resume", json=approval_request(pending).model_dump(mode="json"))
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(str(self.root), response.text)
            self.assertNotIn(pending.workflow.resume_token, response.text)

    def test_import_and_app_creation_do_not_open_workflow_database(self):
        with patch("main.WorkflowPersistence.sqlite", side_effect=AssertionError("must run at startup")):
            app = create_app()
            self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
