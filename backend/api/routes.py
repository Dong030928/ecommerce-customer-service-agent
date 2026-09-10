"""Thin FastAPI routes for HTTP request and response handling."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from yaml import YAMLError

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
    ChatResumeResponse,
    EvalRunRequest,
    EvalRunResponse,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackSubmitResponse,
    TraceEvent,
)
from config.settings import load_agent_capabilities
from evals.runner import EvalRunner, UnknownCaseError
from feedback.attribution import (
    FailureAttributor,
    build_backfilled_case,
    feedback_store,
    safe_feedback_text,
)
from observability.trace import trace_store
from rag.index_cache import get_knowledge_index


def create_router(agent_provider: Any) -> APIRouter:
    """Create API routes and delegate chat execution to the agent provider."""

    router = APIRouter()
    failure_attributor = FailureAttributor()

    def configured_eval_runner() -> EvalRunner:
        return EvalRunner(
            agent_provider(),
            backfilled_cases=feedback_store.list_cases,
        )

    @router.get("/health")
    def health() -> dict[str, str | int]:
        """Return service health and the current project version."""

        index = get_knowledge_index()
        return {
            "status": "ok",
            "version": "0.32.0",
            "rag_index_version": index.version,
            "rag_index_chunks": index.chunk_count,
        }

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        """返回调试后台用于点亮或置灰面板的能力声明。"""

        return load_agent_capabilities()

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """接收最小聊天请求，并返回当前 Agent 的结构化响应。"""

        try:
            return agent_provider().chat(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/chat/resume", response_model=ChatResumeResponse)
    def chat_resume(request: ChatResumeRequest) -> ChatResumeResponse:
        """Resume a paused HITL workflow through the dedicated protocol."""

        try:
            return agent_provider().resume(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/sessions/{session_id}/trace", response_model=list[TraceEvent])
    def session_trace(session_id: str) -> list[TraceEvent]:
        """Return public-safe structured execution events for one session."""

        return trace_store.list(session_id)

    @router.post("/eval/run", response_model=EvalRunResponse)
    def eval_run(request: EvalRunRequest) -> EvalRunResponse:
        """Run the fixed regression suite, or one named case."""

        try:
            return configured_eval_runner().run(request.case_id)
        except UnknownCaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc

    @router.post("/feedback/submit", response_model=FeedbackSubmitResponse)
    def submit_feedback(request: FeedbackRequest) -> FeedbackSubmitResponse:
        """Bind feedback to public evidence and optionally backfill a regression case."""

        trace_events = trace_store.list(request.session_id)
        if not trace_events:
            raise HTTPException(status_code=404, detail="未找到该会话的公开 Trace。")
        runner = configured_eval_runner()
        eval_report = None
        eval_result = None
        base_case = None
        try:
            if request.case_id:
                base_case = next(
                    (case for case in runner.load_cases() if case.case_id == request.case_id),
                    None,
                )
                if base_case is None:
                    raise UnknownCaseError("未找到指定的评测用例。")
                eval_report = runner.run(request.case_id)
                eval_result = eval_report.results[0]
        except UnknownCaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc

        attributions = failure_attributor.attribute(
            feedback=request,
            trace_events=trace_events,
            eval_result=eval_result,
        )
        feedback_id = feedback_store.new_id()
        backfilled_case = (
            build_backfilled_case(
                request,
                attributions,
                feedback_id=feedback_id,
                base_case=base_case,
            )
            if request.rating == "negative"
            else None
        )
        public_case = None if backfilled_case is None else backfilled_case.model_dump(
            exclude={
                "user_message", "runtime_user_id", "runtime_nickname",
                "runtime_member_level", "runtime_risk_level", "runtime_context",
            }
        )
        record = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=request.session_id,
            case_id=request.case_id,
            rating=request.rating,
            user_comment=safe_feedback_text(request.user_comment),
            observed_answer=safe_feedback_text(request.observed_answer),
            trace_event_names=[event.event_type for event in trace_events],
            eval_failure_categories=(eval_result.failure_categories if eval_result else []),
            attributions=attributions,
            backfilled_case=public_case,
        )
        feedback_store.append(record, backfilled_case)
        return FeedbackSubmitResponse(record=record, eval_report=eval_report)

    return router
