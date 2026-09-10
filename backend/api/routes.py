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
    TraceEvent,
)
from config.settings import load_agent_capabilities
from evals.runner import EvalRunner, UnknownCaseError
from observability.trace import trace_store
from rag.index_cache import get_knowledge_index


def create_router(agent_provider: Any) -> APIRouter:
    """Create API routes and delegate chat execution to the agent provider."""

    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str | int]:
        """Return service health and the current project version."""

        index = get_knowledge_index()
        return {
            "status": "ok",
            "version": "0.31.0",
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
            return EvalRunner(agent_provider()).run(request.case_id)
        except UnknownCaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError, YAMLError) as exc:
            raise HTTPException(status_code=500, detail="评测用例配置无法读取或校验失败。") from exc

    return router
