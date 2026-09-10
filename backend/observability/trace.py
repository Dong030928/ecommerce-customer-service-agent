"""Normalize and store public execution traces without hidden reasoning."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from threading import RLock
from typing import Any

from api.schemas import TraceEvent
from config.settings import TRACE_SCHEMA_VERSION


class TraceEventNormalizer:
    """Convert module events into one sanitized, public trace schema."""

    CATEGORY_BY_EVENT = {
        "runtime_context_built": "runtime_context",
        "context_built": "context",
        "route_planned": "planner",
        "rag_pre_retrieved": "rag",
        "tool_started": "tool",
        "tool_finished": "tool",
        "workflow_completed": "workflow",
        "workflow_resumed": "workflow",
        "human_approval_required": "hitl",
        "human_approval_resolved": "hitl",
        "hook_executed": "hook",
        "prompt_security_blocked": "prompt",
        "cost_recorded": "cost",
        "final_answer_generated": "answer",
    }
    BLOCKED_KEYS = {
        "developer_message",
        "frozen_fields",
        "hidden_reasoning",
        "idempotency_key",
        "raw_payload",
        "reasoning_content",
        "requester_id",
        "resume_token",
        "reviewer_id",
        "runtime_user_id",
        "system_prompt",
        "user_id",
    }
    PHONE_PATTERN = re.compile(r"\b1[3-9]\d{9}\b")
    ADDRESS_PATTERN = re.compile(r"(收货地址|地址)\s*[:：]\s*[^,，。;\n]+")
    SECRET_PATTERN = re.compile(
        r"(?i)(?:api[_ -]?key|access[_ -]?token|secret|sk-[A-Za-z0-9_-]{8,})"
    )

    @classmethod
    def normalize(
        cls,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        step: int,
    ) -> TraceEvent:
        safe_payload = cls.sanitize(payload)
        return TraceEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            agent_mode="0.31.0",
            step=step,
            schema_version=TRACE_SCHEMA_VERSION,
            category=cls.CATEGORY_BY_EVENT.get(event_type, "system"),
            stage=cls.stage(event_type, safe_payload),
            name=event_type,
            status=cls.status(event_type, safe_payload),
            target=cls.target(event_type, safe_payload),
            ids={
                "session_id": session_id,
                "workflow_id": safe_payload.get("workflow_id"),
                "order_id": safe_payload.get("order_id"),
                "tool_call_id": safe_payload.get("tool_call_id"),
            },
            summary=cls.summary(event_type, safe_payload),
            signals=cls.signals(event_type, safe_payload),
            safety={
                "public_trace": True,
                "hidden_cot_exposed": False,
                "payload_sanitized": safe_payload != payload,
                "contains_sensitive_raw": False,
            },
            payload=safe_payload,
        )

    @classmethod
    def sanitize(cls, value: Any) -> Any:
        """Recursively remove protected keys and redact sensitive string values."""

        if isinstance(value, dict):
            return {
                key: cls.sanitize(item)
                for key, item in value.items()
                if key not in cls.BLOCKED_KEYS
            }
        if isinstance(value, list):
            return [cls.sanitize(item) for item in value]
        if isinstance(value, str):
            text = cls.PHONE_PATTERN.sub("1**********", value)
            text = cls.ADDRESS_PATTERN.sub(r"\1：[已脱敏地址]", text)
            text = cls.SECRET_PATTERN.sub("[已脱敏密钥]", text)
            text = text.replace("hidden reasoning", "[受保护推理摘要]")
            text = text.replace("隐藏推理", "[受保护推理摘要]")
            return text.replace("系统提示词", "[受保护系统信息]")
        return value

    @staticmethod
    def stage(event_type: str, payload: dict[str, Any]) -> str:
        if event_type.endswith("_started"):
            return "start"
        if event_type.endswith(("_finished", "_completed", "_generated")):
            return "finish"
        if event_type.startswith("rag_"):
            return "retrieval"
        if event_type.startswith("human_approval"):
            return "approval"
        if event_type == "cost_recorded":
            return "cost_summary"
        if event_type == "hook_executed":
            return str(payload.get("hook_type") or "hook")
        return "event"

    @staticmethod
    def status(event_type: str, payload: dict[str, Any]) -> str:
        if payload.get("status"):
            return str(payload["status"])
        if event_type.endswith("_started"):
            return "started"
        if event_type.endswith(("_finished", "_completed", "_generated")):
            return "success"
        if event_type == "human_approval_required":
            return "warning"
        return "recorded"

    @staticmethod
    def target(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": payload.get("target_type")
            or ("tool" if event_type.startswith("tool_") else None),
            "name": payload.get("tool_name")
            or payload.get("workflow_type")
            or payload.get("target_name"),
        }

    @staticmethod
    def summary(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "intent",
            "execution_route",
            "hit_count",
            "retrieval_stage",
            "tool_name",
            "risk_level",
            "needs_human_approval",
            "pending_action",
            "path_type",
            "tool_call_count",
            "degraded",
            "hook_type",
        )
        return {
            "event_type": event_type,
            **{key: payload[key] for key in keys if key in payload},
        }

    @staticmethod
    def signals(event_type: str, payload: dict[str, Any]) -> list[str]:
        signals = [event_type]
        for key in (
            "execution_route",
            "tool_name",
            "workflow_type",
            "pending_action",
            "path_type",
            "hook_type",
        ):
            if payload.get(key):
                signals.append(str(payload[key]))
        if payload.get("needs_human_approval") is True:
            signals.append("needs_human_approval=true")
        return signals


class TraceStore:
    """Thread-safe in-process store shared by chat, resume, and trace routes."""

    def __init__(self) -> None:
        self._events: dict[str, list[TraceEvent]] = {}
        self._lock = RLock()

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._events.clear()
            else:
                self._events.pop(session_id, None)

    def add(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> TraceEvent:
        with self._lock:
            events = self._events.setdefault(session_id, [])
            event = TraceEventNormalizer.normalize(
                session_id=session_id,
                event_type=event_type,
                payload=payload,
                step=len(events) + 1,
            )
            events.append(event)
            return event

    def list(self, session_id: str) -> list[TraceEvent]:
        with self._lock:
            return list(self._events.get(session_id, []))


trace_store = TraceStore()
