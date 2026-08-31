"""Route realtime questions and extract only explicit business identifiers."""

from __future__ import annotations

import re

from api.schemas import Intent
from rag.planning import is_realtime_business_query


ORDER_ID_PATTERN = re.compile(r"\bSO[A-Za-z0-9_-]{6,}\b", re.IGNORECASE)
REFUND_ID_PATTERN = re.compile(r"\bRF-[A-Za-z0-9_-]{3,}\b", re.IGNORECASE)
SKU_PATTERN = re.compile(r"\bSKU-[A-Za-z0-9_-]{3,}\b", re.IGNORECASE)


def extract_order_id(message: str) -> str | None:
    match = ORDER_ID_PATTERN.search(message)
    return match.group(0) if match else None


def extract_refund_request_id(message: str) -> str | None:
    match = REFUND_ID_PATTERN.search(message)
    return match.group(0) if match else None


def extract_sku(message: str) -> str | None:
    match = SKU_PATTERN.search(message)
    return match.group(0).upper() if match else None


def should_route_to_realtime_tool(intent: Intent, message: str) -> bool:
    """Keep stable policies in RAG and route changing facts to tools."""

    if intent == "refund_status_query":
        return True
    if intent == "order_query":
        return is_realtime_business_query(message) or bool(extract_order_id(message))
    if intent != "product_consult":
        return False
    realtime_terms = ["库存", "价格", "多少钱", "还有货", "有没有货", "现价"]
    stable_terms = ["规则", "活动规则", "优惠券", "叠加", "价保规则"]
    return any(term in message for term in realtime_terms) and not any(
        term in message for term in stable_terms
    )


def missing_reference_prompt(intent: Intent, message: str) -> str | None:
    """Lesson-18 boundary: ask for one identifier before model tool selection."""

    if intent == "refund_status_query" and not extract_refund_request_id(message):
        return "请提供退款申请号（例如 RF-1001），我才能查询退款进度。"
    if intent == "order_query" and not extract_order_id(message):
        return "请提供订单号，我才能查询实时订单或物流状态。"
    return None
