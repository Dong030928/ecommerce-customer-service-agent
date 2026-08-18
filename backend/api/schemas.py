"""Pydantic contracts for the public chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Minimal request accepted by the `/chat` endpoint."""

    session_id: str = Field(..., description="当前对话会话 ID")
    # runtime_* values must come from trusted application state rather than user text.
    runtime_user_id: str = Field(..., description="可信调用方确认的用户 ID")
    runtime_nickname: str | None = Field(default=None, description="可信调用方确认的用户昵称")
    runtime_member_level: str | None = Field(default=None, description="可信调用方确认的会员等级")
    runtime_risk_level: str | None = Field(default=None, description="可信调用方确认的风险等级")
    user_message: str = Field(..., description="用户输入的问题")
    runtime_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """`/chat` 返回给调试后台的最小结构化响应。"""

    session_id: str
    answer: str
    session_state: dict[str, Any]
