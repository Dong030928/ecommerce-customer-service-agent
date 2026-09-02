"""Centralized lifecycle governance around controlled tool execution."""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    ChatRequest,
    DegradationState,
    ErrorCategory,
    HookCompletion,
    HookEvent,
    NextAction,
    RiskLevel,
    ToolAction,
    ToolObservation,
    ToolSpec,
)
from tools.runtime_context import public_runtime_context


class HookManager:
    """Collect bounded, sanitized governance events for one chat request."""

    _PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
    _EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
    _SECRET = re.compile(
        r"(?i)(api[_-]?key|secret|token|password|authorization)"
        r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]{6,}"
    )
    _SENSITIVE_KEYS = {
        "api_key",
        "authorization",
        "password",
        "runtime_user_id",
        "secret",
        "service_token",
        "token",
        "user_id",
    }
    _POLLUTION_MARKERS = (
        "忽略规则",
        "忽略之前",
        "跳过审批",
        "直接退款",
        "ignore previous",
        "override system",
        "system prompt",
    )

    def __init__(self) -> None:
        self.events: list[HookEvent] = []
        self._runtime_identity: str | None = None
        self._touched_tools: set[str] = set()
        self._redacted_count = 0
        self._pollution_count = 0
        self._degraded_count = 0
        self._risk_hit_count = 0
        self._tool_count = 0

    def _event(self, **values: Any) -> HookEvent:
        event = HookEvent(sequence=len(self.events) + 1, **values)
        self.events.append(event)
        return event

    def pre_tool_call(
        self,
        action: ToolAction,
        request: ChatRequest,
        spec: ToolSpec | None,
    ) -> HookEvent:
        """Record validation signals before any business read occurs."""

        self._runtime_identity = request.runtime_user_id.strip() or None
        self._tool_count += 1
        self._touched_tools.add(action.tool_name)
        required = spec.required if spec is not None else []
        missing = [field for field in required if not action.arguments.get(field)]
        allowed = set(spec.parameters_schema) if spec is not None else set()
        unexpected = sorted(set(action.arguments) - allowed)
        blocked = spec is None or bool(missing or unexpected) or not spec.read_only
        safe_arguments, args_redacted, args_pollution = self.sanitize(
            action.arguments
        )
        safe_page_context, context_redacted, context_pollution = self.sanitize(
            public_runtime_context(request)
        )
        safe_preview, preview_redacted, preview_pollution = self.sanitize(
            request.user_message[:240]
        )
        redacted = args_redacted or context_redacted or preview_redacted
        pollution = args_pollution or context_pollution or preview_pollution
        self._record_sanitization(redacted, pollution)
        return self._event(
            hook_type="pre_tool_call",
            target_name=action.tool_name,
            action="validate_mcp_tool_arguments",
            result="blocked" if blocked else "allowed",
            reason=(
                "MCP-style 目录提供工具 schema，Hook 统一检查白名单、必填参数、"
                "只读边界和可信身份存在性。"
            ),
            safe_summary={
                "tool": action.tool_name,
                "required": required,
                "missing": missing,
                "unexpected": unexpected,
                "arguments": safe_arguments,
                "runtime_identity_present": bool(request.runtime_user_id.strip()),
                "page_context": safe_page_context,
                "message_preview": safe_preview,
                "read_only": spec.read_only if spec is not None else None,
                "risk_level": spec.risk_level if spec is not None else None,
            },
            redacted=redacted,
            pollution_detected=pollution,
        )

    def post_tool_call(self, observation: ToolObservation) -> ToolObservation:
        """Sanitize the complete Observation before it reaches model/public state."""

        safe_summary, summary_redacted, summary_pollution = self.sanitize(
            observation.summary
        )
        safe_facts, facts_redacted, facts_pollution = self.sanitize(
            observation.facts
        )
        safe_data, data_redacted, data_pollution = self.sanitize(observation.data)
        redacted = summary_redacted or facts_redacted or data_redacted
        pollution = summary_pollution or facts_pollution or data_pollution
        self._record_sanitization(redacted, pollution)
        sanitized = observation.model_copy(
            update={
                "summary": str(safe_summary),
                "facts": safe_facts,
                "data": safe_data,
            }
        )
        self._event(
            hook_type="post_tool_call",
            target_name=observation.tool_name,
            action="sanitize_mcp_tool_observation",
            result="sanitized" if redacted or pollution else "passed",
            reason=(
                "目录工具结果进入模型和公开响应前统一脱敏，并中和外部指令污染。"
            ),
            safe_summary={
                "tool": observation.tool_name,
                "status": observation.status,
                "observation_preview": str(safe_summary)[:180],
                "fact_keys": (
                    sorted(safe_facts) if isinstance(safe_facts, dict) else []
                ),
                "data_keys": (
                    sorted(safe_data) if isinstance(safe_data, dict) else []
                ),
                "omitted_fields": observation.omitted_fields,
                "error_category": observation.error_category,
                "attempts": observation.attempts,
            },
            redacted=redacted,
            pollution_detected=pollution,
        )
        return sanitized

    def on_error(
        self,
        target_name: str,
        error_category: ErrorCategory,
        message: str,
        attempts: int,
    ) -> HookEvent:
        """Normalize an error into a bounded degradation event."""

        self._degraded_count += 1
        safe_message, redacted, pollution = self.sanitize(message)
        self._record_sanitization(redacted, pollution)
        return self._event(
            hook_type="on_error",
            target_name=target_name,
            action="normalize_mcp_tool_error",
            result="degraded",
            reason=(
                "MCP-style 只改变工具来源；工具、模型或检索异常仍被归一为公开"
                "错误类别，不输出原始异常载荷。"
            ),
            safe_summary={
                "error_category": error_category,
                "attempts": max(attempts, 1),
                "message": safe_message,
            },
            redacted=redacted,
            pollution_detected=pollution,
            degraded=True,
        )

    def on_completion(
        self,
        *,
        next_action: NextAction,
        risk_level: RiskLevel,
        degradation: DegradationState,
    ) -> HookCompletion:
        """Emit one public summary without exposing hidden chain-of-thought."""

        if risk_level == "high":
            self._risk_hit_count += 1
        safe_summary = {
            "next_action": next_action,
            "risk_level": risk_level,
            "degraded": degradation.degraded,
            "error_category": degradation.error_category,
            "touched_tools": sorted(self._touched_tools),
            "tool_source": "mcp_catalog",
            "raw_tool_result_exposed": False,
            "hidden_reasoning_exposed": False,
        }
        self._event(
            hook_type="on_completion",
            target_name="chat_request",
            action="summarize_mcp_tool_governance",
            result="completed",
            reason=(
                "请求结束时输出 MCP Tool Use 的有界治理摘要，供调试与测试观察，"
                "不输出隐藏推理链。"
            ),
            safe_summary=safe_summary,
            degraded=degradation.degraded,
        )
        return HookCompletion(
            hook_count=len(self.events),
            tool_count=self._tool_count,
            touched_tools=sorted(self._touched_tools),
            redacted_count=self._redacted_count,
            pollution_count=self._pollution_count,
            degraded_count=self._degraded_count,
            risk_hit_count=self._risk_hit_count,
            safe_summary=safe_summary,
        )

    def sanitize(self, value: Any, *, key: str | None = None) -> tuple[Any, bool, bool]:
        """Recursively redact credentials/PII and neutralize prompt injection text."""

        normalized_key = str(key or "").lower()
        if normalized_key in self._SENSITIVE_KEYS:
            return "[sensitive-redacted]", True, False
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            redacted = False
            pollution = False
            for item_key, item in value.items():
                safe_item, item_redacted, item_pollution = self.sanitize(
                    item,
                    key=str(item_key),
                )
                safe[str(item_key)] = safe_item
                redacted = redacted or item_redacted
                pollution = pollution or item_pollution
            return safe, redacted, pollution
        if isinstance(value, (list, tuple)):
            safe_items: list[Any] = []
            redacted = False
            pollution = False
            for item in value:
                safe_item, item_redacted, item_pollution = self.sanitize(item)
                safe_items.append(safe_item)
                redacted = redacted or item_redacted
                pollution = pollution or item_pollution
            return safe_items, redacted, pollution
        if not isinstance(value, str):
            return value, False, False
        safe = self._PHONE.sub("[phone-redacted]", value)
        safe = self._EMAIL.sub("[email-redacted]", safe)
        if self._runtime_identity and self._runtime_identity in safe:
            safe = safe.replace(self._runtime_identity, "[runtime-identity-redacted]")
        safe = self._SECRET.sub(
            lambda match: f"{match.group(1)}=[secret-redacted]",
            safe,
        )
        redacted = safe != value
        pollution = any(
            marker.lower() in safe.lower() for marker in self._POLLUTION_MARKERS
        )
        if pollution:
            for marker in self._POLLUTION_MARKERS:
                safe = re.sub(
                    re.escape(marker),
                    "[external-instruction-neutralized]",
                    safe,
                    flags=re.IGNORECASE,
                )
        return safe, redacted, pollution

    def _record_sanitization(self, redacted: bool, pollution: bool) -> None:
        if redacted:
            self._redacted_count += 1
        if pollution:
            self._pollution_count += 1
