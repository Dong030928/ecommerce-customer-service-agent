"""Failure-attribution and feedback-backfill regression tests."""

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent
from api.schemas import EvalCaseResult, FeedbackRequest
from feedback.attribution import FailureAttributor, feedback_store
from observability.trace import trace_store


class FailureAttributionTests(unittest.TestCase):
    def setUp(self):
        feedback_store.clear()
        trace_store.clear()

    def tearDown(self):
        feedback_store.clear()
        trace_store.clear()

    @staticmethod
    def request(comment="引用错了，而且工具没调用", answer="没有依据"):
        return FeedbackRequest(
            session_id="feedback-test",
            rating="negative",
            user_comment=comment,
            observed_answer=answer,
            user_message="原始问题",
        )

    def test_attribution_uses_current_eval_failure_categories(self):
        result = EvalCaseResult(
            case_id="failed", session_id="eval-session", passed=False, user_message="问题",
            failure_categories=["tool_path_mismatch", "citation_mismatch", "session_state_mismatch"],
        )
        attributions = FailureAttributor().attribute(
            feedback=self.request(), trace_events=[], eval_result=result,
        )
        self.assertEqual({item.module for item in attributions}, {"Tool", "RAG", "Context"})

    def test_overpromise_is_attributed_to_workflow_and_prompt(self):
        attributions = FailureAttributor().attribute(
            feedback=self.request("客服说不用审批", "已经退款成功并到账"),
            trace_events=[], eval_result=None,
        )
        self.assertEqual({item.module for item in attributions}, {"Workflow", "Prompt"})


class FeedbackApiTests(unittest.TestCase):
    def setUp(self):
        feedback_store.clear()
        trace_store.clear()

    def tearDown(self):
        feedback_store.clear()
        trace_store.clear()

    def client(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI is not installed in this test interpreter")
        from api.routes import create_router

        app = FastAPI()
        app.include_router(create_router(lambda: CustomerServiceAgent()))
        return TestClient(app)

    def create_trace(self, client, session_id="feedback-api"):
        response = client.post("/chat", json={
            "session_id": session_id,
            "runtime_user_id": "U1001",
            "user_message": "你好",
        })
        self.assertEqual(response.status_code, 200, response.text)

    def test_negative_feedback_is_sanitized_and_backfilled(self):
        with self.client() as client:
            self.create_trace(client)
            response = client.post("/feedback/submit", json={
                "session_id": "feedback-api",
                "rating": "negative",
                "user_comment": "回答错误，请复现",
                "observed_answer": "已经退款成功，密钥 sk-abcdefgh12345678",
                "user_message": "你好",
            })
            self.assertEqual(response.status_code, 200, response.text)
            record = response.json()["record"]
            self.assertIn("Workflow", {item["module"] for item in record["attributions"]})
            self.assertNotIn("sk-abcdefgh12345678", record["observed_answer"])
            self.assertNotIn("user_message", record["backfilled_case"])
            self.assertNotIn("runtime_user_id", record["backfilled_case"])

            cases = feedback_store.list_cases()
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].user_message, "你好")
            self.assertEqual(cases[0].source, "feedback_backfill")
            eval_response = client.post("/eval/run", json={"case_id": cases[0].case_id})
            self.assertEqual(eval_response.status_code, 200, eval_response.text)
            self.assertEqual(eval_response.json()["total"], 1)

    def test_positive_feedback_records_without_backfill(self):
        with self.client() as client:
            self.create_trace(client, "positive-session")
            response = client.post("/feedback/submit", json={
                "session_id": "positive-session",
                "rating": "positive",
                "user_comment": "回答正确",
                "observed_answer": "你好，我是电商客服助手。",
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(response.json()["record"]["backfilled_case"])
            self.assertEqual(feedback_store.list_cases(), [])
            self.assertEqual(len(feedback_store.list_records()), 1)

    def test_existing_case_binds_eval_evidence_and_preserves_case_input(self):
        with self.client() as client:
            self.create_trace(client, "linked-session")
            response = client.post("/feedback/submit", json={
                "session_id": "linked-session",
                "case_id": "prompt-injection-trace-boundary",
                "rating": "negative",
                "user_comment": "请加入回归",
                "observed_answer": "不能提供受保护信息",
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["eval_report"]["passed"], 1)
            case = feedback_store.list_cases()[0]
            self.assertIn("hidden reasoning", case.user_message)
            self.assertEqual(case.source, "feedback_backfill")

    def test_unknown_session_case_and_invalid_payload_are_rejected(self):
        with self.client() as client:
            payload = {
                "session_id": "missing",
                "user_comment": "错误",
                "observed_answer": "错误回答",
            }
            self.assertEqual(client.post("/feedback/submit", json=payload).status_code, 404)
            self.create_trace(client, "known-session")
            payload["session_id"] = "known-session"
            payload["case_id"] = "missing-case"
            self.assertEqual(client.post("/feedback/submit", json=payload).status_code, 404)
            payload["extra"] = True
            self.assertEqual(client.post("/feedback/submit", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
