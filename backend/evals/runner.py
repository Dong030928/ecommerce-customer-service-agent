"""Run fixed cases through the configured Agent and grade public evidence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
import yaml

from api.schemas import ChatRequest, ChatResponse, EvalCaseResult, EvalRunResponse
from config.settings import CASES_PATH
from observability.trace import TraceEventNormalizer, TraceStore, trace_store
from safety.prompt_guard import sanitize_text


class ChatAgent(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse: ...


class EvalCase(BaseModel):
    """Reject misspelled expectations rather than silently skipping checks."""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    runtime_user_id: str = "U1001"
    runtime_nickname: str | None = None
    runtime_member_level: str | None = None
    runtime_risk_level: str | None = None
    runtime_context: dict[str, Any] | None = None
    expected_signals: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    forbidden_citations: list[str] = Field(default_factory=list)
    expected_trace_events: list[str] = Field(default_factory=list)
    expected_session_state: list[str] = Field(default_factory=list)
    expected_response: dict[str, Any] = Field(default_factory=dict)
    forbidden_text: list[str] = Field(default_factory=list)


class UnknownCaseError(ValueError):
    """The caller selected a case that is not in the fixed suite."""


class EvalRunner:
    """Use normal Agent dependencies; no mock data or forced offline fallback."""

    def __init__(
        self,
        agent: ChatAgent,
        cases_path: Path = CASES_PATH,
        *,
        traces: TraceStore = trace_store,
    ) -> None:
        self.agent = agent
        self.cases_path = cases_path
        self.traces = traces

    def load_cases(self) -> list[EvalCase]:
        payload = yaml.safe_load(self.cases_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            raise ValueError("评测文件必须包含 cases 列表。")
        cases = [EvalCase.model_validate(item) for item in payload["cases"]]
        ids = [case.case_id for case in cases]
        if not cases or len(ids) != len(set(ids)):
            raise ValueError("评测集不能为空，case_id 不可重复。")
        return cases

    def run(self, case_id: str | None = None) -> EvalRunResponse:
        cases = self.load_cases()
        selected = [case for case in cases if case_id is None or case.case_id == case_id]
        if not selected:
            raise UnknownCaseError("未找到指定的评测用例。")
        run_id = uuid4().hex
        results: list[EvalCaseResult] = []
        for index, case in enumerate(selected):
            session_id = f"eval-{run_id}-{index}"
            result = EvalCaseResult(
                case_id=case.case_id,
                session_id=session_id,
                passed=False,
                user_message=TraceEventNormalizer.sanitize(sanitize_text(case.user_message)[0]),
                expected_signals=case.expected_signals,
            )
            try:
                response = self.agent.chat(ChatRequest(
                    session_id=session_id,
                    **case.model_dump(include={
                        "user_message", "runtime_user_id", "runtime_nickname",
                        "runtime_member_level", "runtime_risk_level", "runtime_context",
                    }),
                ))
                result = self._grade(case, result, response)
            except Exception as exc:
                # A failing service must fail its case, not abort or pass the suite.
                # Do not return provider error bodies, credentials, or stack traces.
                result.error_type = type(exc).__name__
                result.failure_categories = ["execution_error"]
            results.append(result)
        passed = sum(result.passed for result in results)
        return EvalRunResponse(
            run_id=run_id,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            results=results,
            summary={
                "schema_version": "eval_report_v1",
                "failed_cases": [r.case_id for r in results if not r.passed],
                "failure_categories": dict(Counter(
                    category for r in results for category in r.failure_categories
                )),
                "checked_dimensions": [
                    "answer", "tool_calls", "citations", "trace", "session_state", "workflow",
                ],
                "boundary": "固定断言检查配置的真实 Agent 链路；业务与模型用例依赖运行环境，未使用 LLM-as-judge。",
            },
        )

    def _grade(self, case: EvalCase, result: EvalCaseResult, response: ChatResponse) -> EvalCaseResult:
        tools = [record.action.tool_name for record in response.tool_calls]
        citations = [
            value for citation in response.citations
            for value in (citation.citation_id, citation.source_path, citation.source_title,
                          citation.section, citation.chunk_id)
        ]
        trace_events = self.traces.list(result.session_id)
        events = [event.event_type for event in trace_events]
        # Expected signals must come from outcomes, never an echoed input/expectation.
        evidence = " ".join([
            response.answer, *tools, *citations, *events,
            *self._flatten_values(response.workflow.model_dump(
                include={"status", "workflow_type", "pending_action"}
            ) if response.workflow else {}),
        ])
        result.actual_answer = TraceEventNormalizer.sanitize(sanitize_text(response.answer)[0])
        result.actual_tools = TraceEventNormalizer.sanitize(tools)
        result.actual_citations = TraceEventNormalizer.sanitize(citations)
        result.actual_trace_events = events
        result.missing_signals = [s for s in case.expected_signals if s not in evidence]
        result.missing_tools = [t for t in case.expected_tools if t not in tools]
        result.unexpected_tools = [t for t in case.forbidden_tools if t in tools]
        result.missing_citations = [s for s in case.expected_citations if not any(s in c for c in citations)]
        result.forbidden_citation_hits = [s for s in case.forbidden_citations if any(s in c for c in citations)]
        result.missing_trace_events = [e for e in case.expected_trace_events if e not in events]
        result.missing_session_state = [
            r for r in case.expected_session_state
            if not self._matches_session_state(response.session_state, r)
        ]
        response_payload = response.model_dump()
        result.response_mismatches = [
            path for path, expected in case.expected_response.items()
            if not self._matches_value(response_payload, path, expected)
        ]
        # Check original outputs before report redaction so leaks cannot pass.
        output_evidence = " ".join([
            evidence,
            *self._flatten_values([r.observation.model_dump() for r in response.tool_calls]),
            *[c.snippet for c in response.citations],
            *self._flatten_values([e.model_dump(mode="json") for e in trace_events]),
        ])
        result.forbidden_text_hits = [s for s in case.forbidden_text if s in output_evidence]
        checks = {
            "answer_signal_missing": result.missing_signals,
            "tool_path_mismatch": result.missing_tools + result.unexpected_tools,
            "citation_mismatch": result.missing_citations + result.forbidden_citation_hits,
            "trace_event_missing": result.missing_trace_events,
            "session_state_mismatch": result.missing_session_state,
            "response_state_mismatch": result.response_mismatches,
            "forbidden_text_present": result.forbidden_text_hits,
        }
        result.failure_categories = [name for name, failures in checks.items() if failures]
        result.passed = not result.failure_categories
        return result

    @classmethod
    def _flatten_values(cls, value: Any) -> list[str]:
        if isinstance(value, dict):
            return [s for item in value.values() for s in cls._flatten_values(item)]
        if isinstance(value, list):
            return [s for item in value for s in cls._flatten_values(item)]
        return [] if value is None else [str(value)]

    @staticmethod
    def _lookup(payload: dict, path: str) -> tuple[bool, Any]:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    @classmethod
    def _matches_value(cls, payload: dict, path: str, expected: Any) -> bool:
        found, actual = cls._lookup(payload, path)
        return found and type(actual) is type(expected) and actual == expected

    @classmethod
    def _matches_session_state(cls, state: dict, requirement: str) -> bool:
        path, sep, expected = requirement.partition("=")
        found, actual = cls._lookup(state, path.strip().removeprefix("session_state."))
        if not found:
            return False
        if not sep:
            return actual is not None
        normalized = "null" if actual is None else str(actual).lower() if isinstance(actual, bool) else str(actual)
        return normalized == expected.strip()
