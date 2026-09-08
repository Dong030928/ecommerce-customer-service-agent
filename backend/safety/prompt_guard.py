"""Detect and isolate Prompt Injection, secret requests, and private data."""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    ExternalText,
    SafetyDecision,
    SafetyScan,
    ToolObservation,
)


PROMPT_TERMS = (
    "忽略以上",
    "忽略之前",
    "忽略系统",
    "不要遵守",
    "覆盖规则",
    "按我的新规则",
    "直接批准",
    "自动退款",
    "ignore previous",
    "ignore all previous",
    "disregard previous",
)
SECRET_TERMS = (
    "系统提示词",
    "系统指令",
    "developer message",
    "system prompt",
    "hidden reasoning",
    "隐藏推理",
    "内部策略",
    "工具 schema",
    "工具参数",
    "api key",
    "密钥",
)
INJECTION_PATTERNS = (
    r"忽略以上[^。；\n]*",
    r"忽略之前[^。；\n]*",
    r"忽略系统[^。；\n]*",
    r"不要遵守[^。；\n]*",
    r"覆盖规则[^。；\n]*",
    r"按我的新规则[^。；\n]*",
    r"直接批准[^。；\n]*",
    r"自动退款[^。；\n]*",
    r"(?i)ignore (?:all )?previous[^.;\n]*",
    r"(?i)disregard previous[^.;\n]*",
)


def scan_categories(text: str) -> list[str]:
    """Return stable public categories without exposing detection internals."""

    lowered = text.lower()
    categories: list[str] = []
    if any(term in lowered for term in PROMPT_TERMS):
        categories.append("prompt_injection")
    if any(term in lowered for term in SECRET_TERMS):
        categories.append("secret_or_reasoning_request")
    if re.search(r"1[3-9]\d{9}", text) or any(
        term in text for term in ("地址：", "收货地址", "身份证", "银行卡")
    ):
        categories.append("privacy")
    return categories


def sanitize_text(text: str) -> tuple[str, bool]:
    """Redact external instructions and private values before model use."""

    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[已隔离的外部指令]", sanitized)
    sanitized = re.sub(r"1[3-9]\d{9}", "1**********", sanitized)
    sanitized = re.sub(
        r"(身份证(?:号)?[:：]?\s*)\d{17}[\dXx]",
        r"\1[已脱敏证件号]",
        sanitized,
    )
    sanitized = re.sub(
        r"(银行卡(?:号)?[:：]?\s*)(?:\d[ -]?){12,19}",
        r"\1[已脱敏银行卡号]",
        sanitized,
    )
    sanitized = re.sub(
        r"(地址：|收货地址[:：])[^。；\n]+",
        r"\1[已脱敏地址]",
        sanitized,
    )
    for term in SECRET_TERMS:
        sanitized = re.sub(
            re.escape(term),
            "[受保护系统信息]",
            sanitized,
            flags=re.IGNORECASE,
        )
    changed = sanitized != text
    return sanitized, changed


def scan_external_text(text: ExternalText) -> SafetyScan:
    """Scan one source and expose only its sanitized representation."""

    categories = scan_categories(text.content)
    sanitized, _ = sanitize_text(text.content)
    tainted = bool(categories)
    allowed_for_model = "secret_or_reasoning_request" not in categories
    if tainted and allowed_for_model:
        handling = "已标记污染并脱敏，只把安全内容交给模型。"
    elif tainted:
        handling = "涉及系统信息、密钥或隐藏推理，已隔离且不交给模型。"
    else:
        handling = "未发现污染，按外部来源标签进入上下文。"
    return SafetyScan(
        source_type=text.source_type,
        source_id=text.source_id,
        tainted=tainted,
        categories=categories,
        sanitized_content=sanitized,
        allowed_for_model=allowed_for_model,
        handling=handling,
    )


def build_safety_decision(
    user_message: str,
    external_texts: list[ExternalText],
) -> SafetyDecision:
    """Build one request-level decision before routing or model invocation."""

    user_scan = scan_external_text(
        ExternalText(
            source_type="user",
            source_id="user-message",
            content=user_message,
        )
    )
    scans = [user_scan, *(scan_external_text(item) for item in external_texts)]
    blocked = "secret_or_reasoning_request" in user_scan.categories
    return _decision_from_scans(scans, blocked_user_request=blocked)


def extend_safety_decision(
    decision: SafetyDecision,
    external_texts: list[ExternalText],
) -> SafetyDecision:
    """Append tool and RAG scans collected after routing."""

    additions = [
        scan_external_text(item)
        for item in external_texts
    ]
    return _decision_from_scans(
        [*decision.source_scans, *additions],
        blocked_user_request=decision.blocked_user_request,
    )


def build_sanitized_context(decision: SafetyDecision) -> list[str]:
    """Return labeled content that is explicitly allowed to cross the boundary."""

    return [
        (
            f"[{'TAINTED' if scan.tainted else 'CLEAN'}/"
            f"{scan.source_type}/{scan.source_id}] {scan.sanitized_content}"
        )
        for scan in decision.source_scans
        if scan.allowed_for_model
    ]


def sanitize_observation(observation: ToolObservation) -> ToolObservation:
    """Sanitize the Observation before LangChain returns it to the model."""

    return observation.model_copy(
        update={
            "summary": sanitize_text(observation.summary)[0],
            "facts": _sanitize_value(observation.facts),
            "data": _sanitize_value(observation.data),
        }
    )


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)[0]
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def _decision_from_scans(
    scans: list[SafetyScan],
    *,
    blocked_user_request: bool,
) -> SafetyDecision:
    refused = (
        ["system_prompt_or_hidden_reasoning"] if blocked_user_request else []
    )
    public_summary = [
        f"{scan.source_type}:{scan.source_id} 已标记 {', '.join(scan.categories)}。"
        for scan in scans
        if scan.tainted
    ] or ["本轮没有发现需要隔离的外部指令。"]
    return SafetyDecision(
        blocked_user_request=blocked_user_request,
        refused_topics=refused,
        source_scans=scans,
        public_summary=public_summary,
        redaction_applied=any(
            scan.tainted and "未发现污染" not in scan.handling for scan in scans
        ),
    )
