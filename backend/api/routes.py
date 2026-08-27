"""Thin FastAPI routes for HTTP request and response handling."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse
from config.settings import load_agent_capabilities


def create_router(agent_provider: Any) -> APIRouter:
    """Create API routes and delegate chat execution to the agent provider."""

    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        """Return service health and the current project version."""

        return {"status": "ok", "version": "0.5.0"}

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

    return router
