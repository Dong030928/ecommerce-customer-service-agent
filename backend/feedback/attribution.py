"""Bind user feedback to public Trace/Eval evidence and actionable modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from api.schemas import (
    EvalCaseResult,
    FailureAttribution,
    FeedbackRecord,
    FeedbackRequest,
    TraceEvent,
)
from evals.runner import EvalCase
from observability.trace import TraceEventNormalizer
from safety.prompt_guard import sanitize_text


class FailureAttributor:
    """Use deterministic public evidence; attribution is triage, not causality proof."""

    def attribute(
        self,
        *,
        feedback: FeedbackRequest,
        trace_events: list[TraceEvent],
        eval_result: EvalCaseResult | None,
    ) -> list[FailureAttribution]:
        text = f"{feedback.user_comment} {feedback.observed_answer}"
        event_names = [event.event_type for event in trace_events]
        categories = set(eval_result.failure_categories if eval_result else [])
        attributions: list[FailureAttribution] = []

        if any(term in text for term in ["已退款成功", "已经退款成功", "已到账", "不用审批", "跳过审批"]):
            workflow_events = [
                name for name in event_names if "workflow" in name or "approval" in name
            ]
            attributions.extend([
                FailureAttribution(
                    module="Workflow",
                    category="high_risk_boundary",
                    evidence=["反馈中的观察回答出现高风险结果承诺", *workflow_events],
                    suggested_fix="复核售后 Workflow/HITL 边界，禁止绕过审批承诺资金动作完成。",
                ),
                FailureAttribution(
                    module="Prompt",
                    category="overpromise",
                    evidence=["观察回答包含退款完成或到账承诺"],
                    suggested_fix="收紧客服回答边界，只能说明可申请或等待人工审批。",
                ),
            ])
        if "tool_path_mismatch" in categories or any(
            term in text for term in ["没查订单", "没查物流", "工具没调用"]
        ):
            attributions.append(FailureAttribution(
                module="Tool", category="tool_path_mismatch",
                evidence=["Eval 或用户反馈指向工具路径不匹配"],
                suggested_fix="检查路由计划、工具 Schema、参数抽取和执行结果。",
            ))
        if "citation_mismatch" in categories or any(
            term in text for term in ["没有依据", "引用错了", "政策不对"]
        ):
            attributions.append(FailureAttribution(
                module="RAG", category="citation_or_retrieval",
                evidence=["Eval 或用户反馈指向依据缺失或引用错误"],
                suggested_fix="检查检索问题、知识元数据、重排结果和引用期望。",
            ))
        if "session_state_mismatch" in categories or any(
            term in text for term in ["刚才那个", "VIP", "上下文", "串台"]
        ):
            attributions.append(FailureAttribution(
                module="Context", category="context_boundary",
                evidence=["Eval 或用户反馈指向上下文状态或信任边界"],
                suggested_fix="检查 Context Builder、Runtime Context 权威性和 Memory 排除规则。",
            ))
        if categories.intersection({"response_state_mismatch", "trace_event_missing"}):
            attributions.append(FailureAttribution(
                module="Workflow", category="state_or_trace_mismatch",
                evidence=[f"Eval 失败分类：{name}" for name in sorted(categories)],
                suggested_fix="按 Trace 节点复核响应状态、工作流转移和审批暂停信号。",
            ))
        if "answer_signal_missing" in categories and not attributions:
            attributions.append(FailureAttribution(
                module="EvaluationExpectation", category="answer_expectation_gap",
                evidence=["只有回答信号缺失，尚无明确路径退化证据"],
                suggested_fix="先复核用例期望是否过窄，再决定调整 Prompt 或断言。",
            ))
        if not attributions:
            attributions.append(FailureAttribution(
                module="EvaluationExpectation", category="needs_triage",
                evidence=["当前公开反馈、Trace 与 Eval 尚不足以定位模块"],
                suggested_fix="补充可复现输入和具体错误现象，再用 Trace/Eval 缩小范围。",
            ))
        return attributions


def build_backfilled_case(
    feedback: FeedbackRequest,
    attributions: list[FailureAttribution],
    *,
    feedback_id: str,
    base_case: EvalCase | None = None,
) -> EvalCase:
    """Turn negative feedback into an executable in-memory regression case."""

    modules = {item.module for item in attributions}
    source_message = base_case.user_message if base_case else (feedback.user_message or feedback.user_comment)
    payload = base_case.model_dump() if base_case else {
        "case_id": f"feedback-{feedback_id.removeprefix('fb-')}",
        "user_message": source_message,
    }
    payload["case_id"] = f"feedback-{feedback_id.removeprefix('fb-')}"
    payload["user_message"] = source_message
    payload["source"] = "feedback_backfill"
    payload["expected_trace_events"] = list(dict.fromkeys([
        *payload.get("expected_trace_events", []), "cost_recorded",
        *(["rag_pre_retrieved"] if "RAG" in modules else []),
        *(["workflow_completed", "human_approval_required"] if "Workflow" in modules else []),
    ]))
    if "Workflow" in modules:
        payload["expected_signals"] = list(dict.fromkeys([
            *payload.get("expected_signals", []), "人工审批",
        ]))
        payload["expected_session_state"] = list(dict.fromkeys([
            *payload.get("expected_session_state", []),
            "workflow.pending_action=require_human_approval",
        ]))
        payload["forbidden_text"] = list(dict.fromkeys([
            *payload.get("forbidden_text", []), "已退款成功", "已到账",
        ]))
    return EvalCase.model_validate(payload)


class FeedbackStore:
    """Thread-safe feedback workflow store with optional durable JSON storage."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self._lock = RLock()
        self.storage_path = storage_path
        self._records: dict[str, FeedbackRecord] = {}
        self._requests: dict[str, FeedbackRequest] = {}
        self._trace_events: dict[str, list[TraceEvent]] = {}
        self._candidate_cases: dict[str, EvalCase] = {}
        self._approved_cases: dict[str, EvalCase] = {}
        self._load()

    @staticmethod
    def new_id() -> str:
        return f"fb-{uuid4().hex[:12]}"

    def create(
        self,
        record: FeedbackRecord,
        request: FeedbackRequest,
        trace_events: list[TraceEvent],
    ) -> None:
        with self._lock:
            if record.feedback_id in self._records:
                raise ValueError("反馈编号已存在。")
            self._records[record.feedback_id] = record.model_copy(deep=True)
            self._requests[record.feedback_id] = request.model_copy(deep=True)
            self._trace_events[record.feedback_id] = [
                event.model_copy(deep=True) for event in trace_events
            ]
            self._persist()

    def get_record(self, feedback_id: str) -> FeedbackRecord | None:
        with self._lock:
            record = self._records.get(feedback_id)
            return None if record is None else record.model_copy(deep=True)

    def get_request(self, feedback_id: str) -> FeedbackRequest | None:
        with self._lock:
            request = self._requests.get(feedback_id)
            return None if request is None else request.model_copy(deep=True)

    def get_trace_events(self, feedback_id: str) -> list[TraceEvent]:
        with self._lock:
            return [
                event.model_copy(deep=True)
                for event in self._trace_events.get(feedback_id, [])
            ]

    def save_candidate(
        self,
        record: FeedbackRecord,
        candidate: EvalCase,
    ) -> None:
        with self._lock:
            self._require_record(record.feedback_id)
            self._records[record.feedback_id] = record.model_copy(deep=True)
            self._candidate_cases[record.feedback_id] = candidate.model_copy(deep=True)
            self._persist()

    def get_candidate(self, feedback_id: str) -> EvalCase | None:
        with self._lock:
            candidate = self._candidate_cases.get(feedback_id)
            return None if candidate is None else candidate.model_copy(deep=True)

    def finalize(
        self,
        record: FeedbackRecord,
        approved_case: EvalCase | None = None,
    ) -> None:
        with self._lock:
            self._require_record(record.feedback_id)
            self._records[record.feedback_id] = record.model_copy(deep=True)
            if approved_case is not None:
                self._approved_cases[approved_case.case_id] = approved_case.model_copy(deep=True)
                self._candidate_cases[record.feedback_id] = approved_case.model_copy(deep=True)
            self._persist()

    def list_approved_cases(self) -> list[EvalCase]:
        with self._lock:
            return [case.model_copy(deep=True) for case in self._approved_cases.values()]

    def list_cases(self) -> list[EvalCase]:
        """Backward-compatible alias: only approved cases join regression runs."""

        return self.list_approved_cases()

    def list_records(self, status: str | None = None) -> list[FeedbackRecord]:
        with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._records.values()
                if status is None or record.status == status
            ]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._requests.clear()
            self._trace_events.clear()
            self._candidate_cases.clear()
            self._approved_cases.clear()
            self._persist()

    def _require_record(self, feedback_id: str) -> None:
        if feedback_id not in self._records:
            raise KeyError("未找到反馈记录。")

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "feedback_store_v1":
            raise ValueError("反馈存储版本不受支持。")
        self._records = {
            item["feedback_id"]: FeedbackRecord.model_validate(item)
            for item in payload.get("records", [])
        }
        self._requests = {
            feedback_id: FeedbackRequest.model_validate(item)
            for feedback_id, item in payload.get("requests", {}).items()
        }
        self._trace_events = {
            feedback_id: [TraceEvent.model_validate(event) for event in events]
            for feedback_id, events in payload.get("trace_events", {}).items()
        }
        self._candidate_cases = {
            feedback_id: EvalCase.model_validate(item)
            for feedback_id, item in payload.get("candidate_cases", {}).items()
        }
        self._approved_cases = {
            case_id: EvalCase.model_validate(item)
            for case_id, item in payload.get("approved_cases", {}).items()
        }

    def _persist(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "feedback_store_v1",
            "records": [record.model_dump(mode="json") for record in self._records.values()],
            "requests": {
                feedback_id: request.model_dump(mode="json")
                for feedback_id, request in self._requests.items()
            },
            "trace_events": {
                feedback_id: [event.model_dump(mode="json") for event in events]
                for feedback_id, events in self._trace_events.items()
            },
            "candidate_cases": {
                feedback_id: case.model_dump(mode="json")
                for feedback_id, case in self._candidate_cases.items()
            },
            "approved_cases": {
                case_id: case.model_dump(mode="json")
                for case_id, case in self._approved_cases.items()
            },
        }
        temporary = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.storage_path)


def safe_feedback_text(text: str) -> str:
    return TraceEventNormalizer.sanitize(sanitize_text(text)[0])


DEFAULT_FEEDBACK_STORE_PATH = Path(
    os.getenv(
        "AGENT_FEEDBACK_STORE_PATH",
        str(Path(__file__).resolve().parents[2] / ".runtime" / "feedback_store.json"),
    )
)


feedback_store = FeedbackStore(DEFAULT_FEEDBACK_STORE_PATH)
