"""Thin FastAPI routes for HTTP request and response handling."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from yaml import YAMLError

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
    ChatResumeResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalCaseSummary,
    FeedbackCaseConfirmRequest,
    FeedbackCaseReviewRequest,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackSubmitResponse,
    FeedbackStatus,
    TraceEvent,
)
from config.settings import load_agent_capabilities
from evals.runner import EvalRunner, UnknownCaseError
from feedback.attribution import (
    FailureAttributor,
    FeedbackStore,
    build_backfilled_case,
    feedback_store,
    safe_feedback_text,
)
from feedback.case_retrieval import CaseRecommendationService
from observability.trace import trace_store
from rag.index_cache import get_knowledge_index


def create_router(
    agent_provider: Any,
    *,
    feedback_repository: FeedbackStore | None = None,
    case_recommender: CaseRecommendationService | None = None,
) -> APIRouter:
    """Create API routes and delegate chat execution to the agent provider."""

    router = APIRouter()
    repository = feedback_repository or feedback_store
    recommender = case_recommender or CaseRecommendationService()
    failure_attributor = FailureAttributor()

    def configured_eval_runner() -> EvalRunner:
        return EvalRunner(
            agent_provider(),
            backfilled_cases=repository.list_approved_cases,
        )

    @router.get("/health")
    def health() -> dict[str, str | int]:
        """Return service health and the current project version."""

        index = get_knowledge_index()
        return {
            "status": "ok",
            "version": "0.34.0",
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

    @router.get("/eval/cases", response_model=list[EvalCaseSummary])
    def eval_cases() -> list[EvalCaseSummary]:
        """List fixed and approved cases for trusted manual case selection."""

        try:
            cases = configured_eval_runner().load_cases()
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc
        return [
            EvalCaseSummary(
                case_id=case.case_id,
                scenario_summary=safe_feedback_text(case.user_message),
                source=case.source,
            )
            for case in cases
        ]

    @router.post("/feedback/submit", response_model=FeedbackSubmitResponse)
    def submit_feedback(request: FeedbackRequest) -> FeedbackSubmitResponse:
        """Record feedback and recommend similar cases without running them yet."""

        trace_events = trace_store.list(request.session_id)
        if not trace_events:
            raise HTTPException(status_code=404, detail="未找到该会话的公开 Trace。")
        safe_request = request.model_copy(update={
            "user_comment": safe_feedback_text(request.user_comment),
            "observed_answer": safe_feedback_text(request.observed_answer),
            "user_message": (
                safe_feedback_text(request.user_message)
                if request.user_message is not None
                else None
            ),
        })
        runner = configured_eval_runner()
        recommendations = []
        recommendation_error = None
        try:
            available_cases = runner.load_cases() if request.rating == "negative" else []
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc
        try:
            if available_cases:
                recommendations = recommender.recommend(safe_request, available_cases)
        except (RuntimeError, ValueError, httpx.HTTPError):
            recommendation_error = "Case 向量推荐暂不可用，请由审核人员手动选择用例。"
        feedback_id = repository.new_id()
        now = datetime.now(timezone.utc)
        record = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=request.session_id,
            message_id=request.message_id,
            rating=request.rating,
            status=("awaiting_case_confirmation" if request.rating == "negative" else "recorded"),
            user_comment=safe_request.user_comment,
            observed_answer=safe_request.observed_answer,
            trace_event_names=[event.event_type for event in trace_events],
            recommendations=recommendations,
            recommendation_error=recommendation_error,
            created_at=now,
            updated_at=now,
        )
        repository.create(record, safe_request, trace_events)
        return FeedbackSubmitResponse(record=record)

    @router.get("/feedback", response_model=list[FeedbackRecord])
    def list_feedback(status: FeedbackStatus | None = None) -> list[FeedbackRecord]:
        """List the operator inbox, optionally filtered by workflow status."""

        return repository.list_records(status)

    @router.get("/feedback/{feedback_id}", response_model=FeedbackRecord)
    def get_feedback(feedback_id: str) -> FeedbackRecord:
        """Return one feedback workflow record for a trusted review console."""

        record = repository.get_record(feedback_id)
        if record is None:
            raise HTTPException(status_code=404, detail="未找到反馈记录。")
        return record

    @router.post(
        "/feedback/{feedback_id}/case-confirm",
        response_model=FeedbackSubmitResponse,
    )
    def confirm_feedback_case(
        feedback_id: str,
        request: FeedbackCaseConfirmRequest,
    ) -> FeedbackSubmitResponse:
        """Confirm one recommended case, or explicitly declare a new scenario."""

        record = repository.get_record(feedback_id)
        stored_request = repository.get_request(feedback_id)
        trace_events = repository.get_trace_events(feedback_id)
        if record is None or stored_request is None:
            raise HTTPException(status_code=404, detail="未找到反馈记录。")
        if record.status != "awaiting_case_confirmation":
            raise HTTPException(status_code=409, detail="反馈当前状态不允许确认 Case。")
        if (request.case_id is None) == (not request.no_matching_case):
            raise HTTPException(status_code=422, detail="必须选择一个 Case，或明确标记没有匹配 Case。")

        runner = configured_eval_runner()
        base_case = None
        eval_report = None
        eval_result = None
        try:
            if request.case_id is not None:
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
            feedback=stored_request,
            trace_events=trace_events,
            eval_result=eval_result,
        )
        candidate = build_backfilled_case(
            stored_request,
            attributions,
            feedback_id=feedback_id,
            base_case=base_case,
        )
        updated = record.model_copy(update={
            "case_id": request.case_id,
            "status": "pending_case_review",
            "eval_failure_categories": eval_result.failure_categories if eval_result else [],
            "attributions": attributions,
            "backfilled_case": _public_case(candidate),
            "reviewer_id": request.reviewer_id,
            "reviewer_note": safe_feedback_text(request.reviewer_note or "") or None,
            "updated_at": datetime.now(timezone.utc),
        })
        repository.save_candidate(updated, candidate)
        return FeedbackSubmitResponse(record=updated, eval_report=eval_report)

    @router.post(
        "/feedback/{feedback_id}/review",
        response_model=FeedbackRecord,
    )
    def review_feedback_case(
        feedback_id: str,
        request: FeedbackCaseReviewRequest,
    ) -> FeedbackRecord:
        """Approve, reject, or merge a candidate before it joins regression."""

        record = repository.get_record(feedback_id)
        candidate = repository.get_candidate(feedback_id)
        if record is None:
            raise HTTPException(status_code=404, detail="未找到反馈记录。")
        if record.status != "pending_case_review" or candidate is None:
            raise HTTPException(status_code=409, detail="反馈当前没有可审核的候选 Case。")

        try:
            existing_case_ids = {
                case.case_id for case in configured_eval_runner().load_cases()
            }
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc
        approved_case = None
        merged_into = None
        if request.decision == "approved":
            if request.target_case_id is not None:
                raise HTTPException(status_code=422, detail="批准新 Case 时不能指定合并目标。")
            updates = (
                request.case_updates.model_dump(exclude_none=True)
                if request.case_updates is not None
                else {}
            )
            approved_case = candidate.model_copy(update=updates)
            approved_case = type(candidate).model_validate(approved_case.model_dump())
            next_status = "approved"
        elif request.decision == "merged":
            if request.target_case_id is None or request.target_case_id not in existing_case_ids:
                raise HTTPException(status_code=422, detail="合并时必须指定一个已存在的目标 Case。")
            merged_into = request.target_case_id
            next_status = "merged"
        else:
            if request.target_case_id is not None or request.case_updates is not None:
                raise HTTPException(status_code=422, detail="拒绝 Case 时不能提交合并目标或修改内容。")
            next_status = "rejected"

        updated = record.model_copy(update={
            "status": next_status,
            "backfilled_case": _public_case(approved_case or candidate),
            "reviewer_id": request.reviewer_id,
            "reviewer_note": safe_feedback_text(request.reviewer_note or "") or None,
            "merged_into_case_id": merged_into,
            "updated_at": datetime.now(timezone.utc),
        })
        repository.finalize(updated, approved_case)
        return updated

    return router


def _public_case(case: Any) -> dict[str, Any]:
    """Expose reviewable assertions without leaking incident or runtime context."""

    return case.model_dump(exclude={
        "user_message", "runtime_user_id", "runtime_nickname",
        "runtime_member_level", "runtime_risk_level", "runtime_context",
    })
