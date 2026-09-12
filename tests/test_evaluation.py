"""Fixed-case evaluator tests; business/model doubles are test-only."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent
from api.schemas import ChatRequest, Citation, EvalCaseResult, ToolAction, ToolCallRecord, ToolObservation
from evals.runner import EvalCase, EvalRunner, UnknownCaseError
from observability.trace import trace_store
from rag.index_cache import reset_index_and_cache
from rag.query_rewrite import rewrite_retrieval_query
from policies.after_sale_policy import AfterSalePolicyService
from tools.tool_runtime import ToolRuntime
from tools.planning import extract_order_id
from tools.tool_calling import ToolCallingOutcome
import test_langgraph_workflow as workflow_fixtures


class EvaluationEmbeddingFixture(workflow_fixtures.WorkflowEmbeddingClient):
    """Explicit small vector fixture distinguishing unshipped refund from returns."""

    @staticmethod
    def _vector(text):
        return [
            3.0 if any(term in text for term in ("未发货", "没发货", "待发货")) else 0.0,
            1.0 if "退款" in text else 0.0,
            1.0,
        ]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        trace_store.clear()
        reset_index_and_cache()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(trace_store.clear)
        self.addCleanup(reset_index_and_cache)
        self.path = Path(self.temp.name) / "cases.yml"

    def runner(self, cases, agent=None):
        self.path.write_text(yaml.safe_dump({"cases": cases}, allow_unicode=True), encoding="utf-8")
        return EvalRunner(agent or CustomerServiceAgent(), self.path)

    def test_real_security_case_and_unique_sessions(self):
        runner = EvalRunner(CustomerServiceAgent())
        first = runner.run("prompt-injection-trace-boundary")
        second = runner.run("prompt-injection-trace-boundary")
        self.assertEqual((first.total, first.passed, first.failed), (1, 1, 0), first.model_dump())
        self.assertEqual(first.summary["schema_version"], "eval_report_v1")
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.results[0].session_id, second.results[0].session_id)

    def test_refund_case_uses_actual_workflow_with_test_business_and_embedding(self):
        client = workflow_fixtures.WorkflowEcommerceClient()
        agent = CustomerServiceAgent(
            embedding_client=EvaluationEmbeddingFixture(), answer_api_key="",
            after_sale_policy_service=AfterSalePolicyService(ToolRuntime(client)),
        )
        result = EvalRunner(agent).run("unshipped-refund-hitl")
        self.assertEqual(result.passed, 1, result.model_dump())
        self.assertIn("get_order", client.calls)
        self.assertIn("get_logistics", client.calls)
        self.assertIn("human_approval_required", result.results[0].actual_trace_events)

    def test_logistics_case_with_explicit_model_and_business_doubles(self):
        client = workflow_fixtures.WorkflowEcommerceClient()
        client.order_status = "SHIPPED"
        client.logistics_status = "IN_TRANSIT"
        runtime = ToolRuntime(client)

        def tool_run(request, intent, clarification_plan=None, hooks=None, allowed_tool_names=None):
            self.assertIn("get_order_logistics", allowed_tool_names)
            action = ToolAction(tool_name="get_order_logistics",
                                arguments={"order_id": extract_order_id(request.user_message)}, reason="测试物流查询")
            observation = runtime.execute(action, request, hooks)
            return ToolCallingOutcome(
                answer=observation.summary, tool_calls=[ToolCallRecord(action=action, observation=observation)],
                state={"answer_source": "sanitized_tool_observation"}, used_model=False,
            )

        agent = CustomerServiceAgent(tool_calling_service=Mock(run=tool_run))
        report = EvalRunner(agent).run("order-logistics-tool-path")
        self.assertEqual(report.passed, 1, report.model_dump())
        self.assertEqual(client.calls, ["get_order", "get_logistics"])
        self.assertEqual(report.results[0].actual_citations, [])

    def test_resume_case_reuses_real_checkpoint_and_grades_idempotency(self):
        client = workflow_fixtures.WorkflowEcommerceClient()
        agent = CustomerServiceAgent(
            embedding_client=EvaluationEmbeddingFixture(),
            answer_api_key="",
            after_sale_policy_service=AfterSalePolicyService(ToolRuntime(client)),
        )
        runner = self.runner(
            [
                {
                    "case_id": "resume-approved-idempotent",
                    "case_type": "resume",
                    "user_message": (
                        f"订单 {workflow_fixtures.ORDER_ID} 还没发货，直接退款"
                    ),
                    "repeat_resume": True,
                    "expected_trace_events": [
                        "human_approval_required",
                        "workflow_resumed",
                        "human_approval_resolved",
                    ],
                    "expected_session_state": [
                        "resume_status=completed",
                        "resume_result.accepted=true",
                        "resume_result.idempotent_replay=true",
                        "business_recheck.passed=true",
                    ],
                    "forbidden_text": ["已到账"],
                }
            ],
            agent,
        )

        report = runner.run()

        self.assertEqual(
            (report.total, report.passed, report.failed),
            (1, 1, 0),
            report.model_dump(),
        )

    def test_resume_case_can_assert_invalid_token_boundary(self):
        client = workflow_fixtures.WorkflowEcommerceClient()
        agent = CustomerServiceAgent(
            embedding_client=EvaluationEmbeddingFixture(),
            answer_api_key="",
            after_sale_policy_service=AfterSalePolicyService(ToolRuntime(client)),
        )
        runner = self.runner(
            [
                {
                    "case_id": "resume-invalid-token",
                    "case_type": "resume",
                    "user_message": (
                        f"订单 {workflow_fixtures.ORDER_ID} 还没发货，直接退款"
                    ),
                    "resume_token_override": "invalid-token",
                    "expected_session_state": [
                        "resume_status=blocked",
                        "resume_result.accepted=false",
                        "resume_result.reason=resume_token 不匹配，不能恢复这个审批流程。",
                    ],
                    "forbidden_text": ["审批通过", "已到账"],
                }
            ],
            agent,
        )

        report = runner.run()

        self.assertEqual(
            (report.total, report.passed, report.failed),
            (1, 1, 0),
            report.model_dump(),
        )

    def test_all_dimensions_fail_with_explicit_categories(self):
        response = CustomerServiceAgent().chat(ChatRequest(
            session_id="negative-fixture", runtime_user_id="U1001", user_message="你好"
        ))
        response.answer = "已到账 sk-abcdefgh12345678"
        response.tool_calls = [ToolCallRecord(
            action=ToolAction(tool_name="get_order_status", arguments={}, reason="测试"),
            observation=ToolObservation(tool_name="get_order_status", status="success", summary="测试"),
        )]
        response.citations = [Citation(
            citation_id="C1", source_title="错误来源", source_path="wrong.md",
            section="错误章节", chunk_id="wrong-chunk", score=1.0, snippet="示例",
        )]
        case = EvalCase(
            case_id="negative", user_message="输入里的预期词不算输出", expected_signals=["预期词"],
            expected_tools=["get_order_logistics"], forbidden_tools=["get_order_status"],
            expected_citations=["correct-chunk"], forbidden_citations=["wrong.md"],
            expected_trace_events=["human_approval_required"],
            expected_session_state=["absent=null"], expected_response={"needs_human_approval": True},
            forbidden_text=["已到账", "sk-"],
        )
        result = EvalRunner(CustomerServiceAgent())._grade(
            case, EvalCaseResult(case_id=case.case_id, session_id="no-traces", passed=False,
                                 user_message=case.user_message), response,
        )
        self.assertFalse(result.passed)
        self.assertEqual(set(result.failure_categories), {
            "answer_signal_missing", "tool_path_mismatch", "citation_mismatch", "trace_event_missing",
            "session_state_mismatch", "response_state_mismatch", "forbidden_text_present",
        })
        self.assertEqual(result.forbidden_text_hits, ["已到账", "sk-"])
        self.assertNotIn("sk-abcdefgh12345678", result.actual_answer)

    def test_forbidden_text_in_citation_is_checked_before_report_redaction(self):
        response = CustomerServiceAgent().chat(ChatRequest(
            session_id="citation-fixture", runtime_user_id="U1001", user_message="你好"
        ))
        response.citations = [Citation(
            citation_id="C1", source_title="规则", source_path="rules.md", section="政策",
            chunk_id="rules", score=1.0, snippet="不应暴露的 SYSTEM_PROMPT",
        )]
        runner = self.runner([{"case_id": "leak", "user_message": "你好", "forbidden_text": ["SYSTEM_PROMPT"]}],
                             Mock(chat=Mock(return_value=response)))
        self.assertEqual(runner.run().results[0].forbidden_text_hits, ["SYSTEM_PROMPT"])

    def test_execution_failure_isolated_and_error_body_not_returned(self):
        real_agent = CustomerServiceAgent()
        def chat(request):
            if request.user_message == "失败":
                raise RuntimeError("private-provider-error sk-abcdefgh12345678")
            return real_agent.chat(request)
        runner = self.runner([
            {"case_id": "failed", "user_message": "失败"},
            {"case_id": "ok", "user_message": "你好", "expected_trace_events": ["cost_recorded"]},
        ], Mock(chat=chat))
        report = runner.run()
        self.assertEqual((report.total, report.passed, report.failed), (2, 1, 1))
        self.assertEqual(report.results[0].error_type, "RuntimeError")
        self.assertEqual(report.summary["failure_categories"], {"execution_error": 1})
        self.assertNotIn("private-provider-error", report.model_dump_json())

    def test_unknown_case_does_not_execute_agent(self):
        agent = Mock()
        runner = self.runner([{"case_id": "ok", "user_message": "你好"}], agent)
        for unknown in ("missing", ""):
            with self.assertRaises(UnknownCaseError):
                runner.run(unknown)
        agent.chat.assert_not_called()

    def test_invalid_suites_fail_before_execution(self):
        base = {"case_id": "ok", "user_message": "你好"}
        for cases in ([], [base, base], [{**base, "expected_tool": ["misspelled"]}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.runner(cases).run()
        self.path.write_text("cases: [", encoding="utf-8")
        with self.assertRaises(yaml.YAMLError):
            EvalRunner(CustomerServiceAgent(), self.path).run()

    def test_state_checks_preserve_missing_null_and_boolean_distinctions(self):
        self.assertFalse(EvalRunner._matches_session_state({}, "missing=null"))
        self.assertTrue(EvalRunner._matches_session_state({"a": None}, "a=null"))
        self.assertTrue(EvalRunner._matches_session_state({"a": False}, "session_state.a=false"))
        self.assertFalse(EvalRunner._matches_value({"a": 1}, "a", True))
        self.assertFalse(EvalRunner._matches_value({}, "a", None))

    def test_unshipped_refund_query_does_not_expand_into_received_return(self):
        query = rewrite_retrieval_query("订单还没发货，能退款吗", "refund_request")
        self.assertIn("人工审批", query.added_terms)
        self.assertNotIn("签收时间", query.added_terms)
        self.assertNotIn("退货条件", query.added_terms)
        received = rewrite_retrieval_query("签收后怎么退货", "refund_request")
        self.assertIn("签收时间", received.added_terms)


class EvaluationApiTests(unittest.TestCase):
    def test_endpoint_reports_and_validation(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI is not installed in this test interpreter")
        from api.routes import create_router

        agent = CustomerServiceAgent()
        app = FastAPI()
        app.include_router(create_router(lambda: agent))
        self.addCleanup(trace_store.clear)
        with TestClient(app) as client:
            response = client.post("/eval/run", json={"case_id": "prompt-injection-trace-boundary"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["passed"], 1, response.text)
            self.assertEqual(client.post("/eval/run", json={"case_id": "missing"}).status_code, 404)
            self.assertEqual(client.post("/eval/run", json={"case_id": []}).status_code, 422)
            self.assertEqual(client.post("/eval/run", json={"caseId": "misspelled"}).status_code, 422)
            with patch("api.routes.EvalRunner.run", side_effect=yaml.YAMLError("private-path")):
                response = client.post("/eval/run", json={})
                self.assertEqual(response.status_code, 500)
                self.assertNotIn("private-path", response.text)


if __name__ == "__main__":
    unittest.main()
