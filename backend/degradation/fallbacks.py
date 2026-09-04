"""Deterministic degradation policies for model and tool failures."""

from __future__ import annotations

import re

from api.schemas import DegradationState, ErrorCategory, ToolCallRecord


ERROR_CODE_CATEGORIES: dict[str, ErrorCategory] = {
    "business_timeout": "timeout",
    "tool_not_allowed": "validation_error",
    "tool_arguments_missing": "validation_error",
    "tool_arguments_not_allowed": "validation_error",
    "order_id_invalid": "validation_error",
    "refund_request_id_invalid": "validation_error",
    "order_month_invalid": "validation_error",
    "product_ambiguous": "validation_error",
    "business_fact_not_found": "not_found",
    "product_not_found": "not_found",
    "business_access_denied": "forbidden",
    "runtime_identity_missing": "forbidden",
    "service_auth_missing": "forbidden",
    "business_request_failed": "business_error",
    "business_service_unavailable": "system_error",
    "business_response_invalid": "system_error",
    "tool_not_executed": "system_error",
    "observation_rule_missing": "system_error",
    "high_risk_write_blocked": "high_risk_write_blocked",
}

FALLBACK_MESSAGES: dict[ErrorCategory, str] = {
    "none": "",
    "timeout": "实时业务查询超时了，系统已完成有限重试，仍未取得可信结果。请稍后再试。",
    "validation_error": "这次查询参数不完整或格式不正确，请补充准确的业务编号后再试。",
    "not_found": "业务系统没有查到对应记录，请核对编号是否正确以及记录是否属于当前账号。",
    "forbidden": "当前身份无权读取这条业务记录，或服务身份尚未配置，无法继续查询。",
    "business_error": "业务系统拒绝了本次请求，请核对业务条件后再试。",
    "model_unavailable": "智能客服模型当前不可用，无法安全完成实时工具规划，请稍后再试。",
    "system_error": "实时业务服务暂时不可用，当前没有取得可信结果，请稍后再试。",
    "high_risk_write_blocked": (
        "退款、取消订单或赔付属于高风险写操作，本 Agent 不会直接执行。"
        "请转人工客服核验订单、金额和责任后处理。"
    ),
}

_HIGH_RISK_WRITE_PATTERN = re.compile(
    r"(?:直接|马上|立刻|立即|赶紧|帮我|给我|我要|替我|现在)"
    r"[^，。！？]{0,10}(?:退款|退钱|退货|取消订单|赔偿|赔付|补偿)"
    r"|(?:申请|办理|执行|发起)(?:退款|退货|取消订单|赔偿|赔付|补偿)"
    r"|(?:七天|7天)无理由退货"
    r"|(?:退款|退货|取消订单)(?:吧|！|。|$)"
)
_STATUS_QUERY_PATTERN = re.compile(
    r"(?:退款|退货|赔付|补偿)(?:进度|状态|到哪|到账|什么时候到)"
)
_INFORMATION_QUERY_PATTERN = re.compile(
    r"(?:如何|怎么|怎样|流程|条件|规则|政策|能否|是否可以|需要什么)"
    r"[^，。！？]{0,12}(?:退款|退货|取消订单|赔偿|赔付|补偿)"
)
_NEGATED_WRITE_PATTERN = re.compile(
    r"(?:不要|不用|无需|不想|暂不|不是要)[^，。！？]{0,8}"
    r"(?:退款|退钱|退货|取消订单|赔偿|赔付|补偿)"
)


def classify_error_code(error_code: str | None) -> ErrorCategory:
    """Map implementation-specific error codes to stable public categories."""

    if not error_code:
        return "none"
    return ERROR_CODE_CATEGORIES.get(error_code, "system_error")


def fallback_message(category: ErrorCategory) -> str:
    """Return a safe answer that never claims an operation succeeded."""

    return FALLBACK_MESSAGES[category]


def is_high_risk_write_request(message: str) -> bool:
    """Detect explicit write-operation requests without blocking status queries."""

    normalized = "".join(message.split())
    if (
        _STATUS_QUERY_PATTERN.search(normalized)
        or _INFORMATION_QUERY_PATTERN.search(normalized)
        or _NEGATED_WRITE_PATTERN.search(normalized)
    ):
        return False
    return _HIGH_RISK_WRITE_PATTERN.search(normalized) is not None


def degradation_from_tool_records(
    records: list[ToolCallRecord],
    *,
    model_error: str | None = None,
) -> DegradationState:
    """Summarize retry and failure signals from public-safe tool records."""

    retry_count = sum(max(record.attempts - 1, 0) for record in records)
    failed = [
        record
        for record in records
        if record.observation.status != "success"
        and record.observation.next_action != "ask_clarification"
    ]
    if failed:
        category = failed[-1].observation.error_category
        return DegradationState(
            degraded=True,
            error_category=category,
            retry_count=retry_count,
            fallback_used=True,
            reason=failed[-1].observation.error_code or category,
        )
    if model_error:
        return DegradationState(
            degraded=True,
            error_category="model_unavailable",
            retry_count=retry_count,
            fallback_used=True,
            reason=model_error,
        )
    return DegradationState(retry_count=retry_count)


def high_risk_degradation() -> DegradationState:
    """Build the boundary state for an intentionally blocked write request."""

    return DegradationState(
        degraded=True,
        error_category="high_risk_write_blocked",
        fallback_used=True,
        reason="high_risk_write_blocked",
    )
