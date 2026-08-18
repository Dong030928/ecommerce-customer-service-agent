"""Pydantic contracts for the public chat API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReasoningView = Literal["default", "off", "summary", "teaching"]
IntentSource = Literal["rules", "classifier", "rules_fallback"]
Intent = Literal[
    "general_chat",
    "promotion_consult",
    "product_consult",
    "order_query",
    "refund_request",
    "complaint",
    "unknown",
]


class ChatRequest(BaseModel):
    """Minimal request accepted by the `/chat` endpoint."""

    session_id: str = Field(..., description="当前对话会话 ID")
    # runtime_* values must come from trusted application state rather than user text.
    runtime_user_id: str = Field(..., description="可信调用方确认的用户 ID")
    runtime_nickname: str | None = Field(default=None, description="可信调用方确认的用户昵称")
    runtime_member_level: str | None = Field(default=None, description="可信调用方确认的会员等级")
    runtime_risk_level: str | None = Field(default=None, description="可信调用方确认的风险等级")
    user_message: str = Field(..., description="用户输入的问题")
    reasoning_view: ReasoningView = "default"
    debug: bool = True
    runtime_context: dict[str, Any] | None = None


class IntentResult(BaseModel):
    """Stable, machine-readable classification result for downstream routing."""

    intent: Intent
    source: IntentSource
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)
    explanation: str


class ChatResponse(BaseModel):
    """`/chat` 返回给调试后台的最小结构化响应。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    reasoning_summary: list[str]
    session_state: dict[str, Any]


ChatRequest.model_rebuild(_types_namespace={"Any": Any, "ReasoningView": ReasoningView})
IntentResult.model_rebuild(_types_namespace={"Intent": Intent, "IntentSource": IntentSource})
ChatResponse.model_rebuild(_types_namespace={"Any": Any, "Intent": Intent, "IntentResult": IntentResult})
