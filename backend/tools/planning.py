"""Route realtime questions and extract only explicit business identifiers."""

from __future__ import annotations

import re
from typing import Any

from api.schemas import (
    ChatRequest,
    ClarificationCandidate,
    ClarificationPlan,
    ClarificationRequest,
    Intent,
    ToolCallRecord,
)
from rag.planning import is_realtime_business_query
from tools.contracts import TOOL_SPECS
from tools.runtime_context import contextual_order_id, order_candidates, runtime_context


ORDER_ID_PATTERN = re.compile(r"\bSO[A-Za-z0-9_-]{6,}\b", re.IGNORECASE)
REFUND_ID_PATTERN = re.compile(r"\bRF-[A-Za-z0-9_-]{3,}\b", re.IGNORECASE)
SKU_PATTERN = re.compile(r"\bSKU-[A-Za-z0-9_-]{3,}\b", re.IGNORECASE)
MONTH_PATTERN = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*月")
PRODUCT_REFERENCES = ["耳机", "充电器", "音箱"]
PRODUCT_REALTIME_TERMS = ["库存", "价格", "多少钱", "还有货", "有没有货", "现价"]
PRODUCT_KNOWLEDGE_TERMS = [
    "推荐",
    "适合",
    "通勤",
    "差旅",
    "续航",
    "规则",
    "活动",
    "优惠",
    "会员价",
    "优惠券",
    "叠加",
]
CLARIFICATION_FIELD_TERMS = {
    "order_id": ["订单", "订单号"],
    "refund_request_id": ["退款", "申请号"],
    "sku": ["商品", "商品名", "SKU"],
    "month": ["月", "月份"],
}
FORBIDDEN_CLARIFICATION_TERMS = ["密码", "验证码", "银行卡", "身份证", "密钥", "token"]


def extract_order_id(message: str) -> str | None:
    match = ORDER_ID_PATTERN.search(message)
    return match.group(0) if match else None


def extract_refund_request_id(message: str) -> str | None:
    match = REFUND_ID_PATTERN.search(message)
    return match.group(0) if match else None


def extract_sku(message: str) -> str | None:
    match = SKU_PATTERN.search(message)
    return match.group(0).upper() if match else None


def extract_month(message: str) -> int | None:
    match = MONTH_PATTERN.search(message)
    return int(match.group(1)) if match else None


def extract_product_reference(message: str) -> str | None:
    sku = extract_sku(message)
    if sku:
        return sku
    return next((term for term in PRODUCT_REFERENCES if term in message), None)


def is_product_tool_rag_query(intent: Intent, message: str) -> bool:
    """Route mixed current-fact and stable-knowledge questions to both sources."""

    if intent != "product_consult" or extract_product_reference(message) is None:
        return False
    return any(term in message for term in PRODUCT_REALTIME_TERMS) and any(
        term in message for term in PRODUCT_KNOWLEDGE_TERMS
    )


def select_realtime_tool(intent: Intent, message: str) -> str | None:
    if intent == "refund_status_query":
        return "get_refund_status"
    if intent == "product_consult":
        return "get_product_inventory"
    if intent != "order_query":
        return None
    if not extract_order_id(message) and extract_month(message) is not None:
        return "search_current_user_orders"
    if any(term in message for term in ["物流", "快递", "运单", "到哪", "发货"]):
        return "get_order_logistics"
    return "get_order_status"


def _known_arguments(
    request: ChatRequest,
    intent: Intent,
    tool_name: str | None,
) -> dict[str, Any]:
    if tool_name in {"get_order_status", "get_order_logistics"}:
        order_id = extract_order_id(request.user_message)
        if not order_id:
            contextual = contextual_order_id(request)
            if contextual and ORDER_ID_PATTERN.fullmatch(contextual):
                order_id = contextual
        return {"order_id": order_id} if order_id else {}
    if tool_name == "get_refund_status":
        context_refund_id = str(
            runtime_context(request).get("relatedAfterSaleNo") or ""
        ).strip()
        refund_id = extract_refund_request_id(request.user_message)
        if not refund_id and REFUND_ID_PATTERN.fullmatch(context_refund_id):
            refund_id = context_refund_id
        return {"refund_request_id": refund_id} if refund_id else {}
    if tool_name == "get_product_inventory":
        product = extract_product_reference(request.user_message)
        return {"sku": product} if product else {}
    if tool_name == "search_current_user_orders":
        month = extract_month(request.user_message)
        return {"month": month} if month is not None else {}
    return {}


def _missing_required(
    tool_name: str | None,
    arguments: dict[str, Any],
) -> list[str]:
    if tool_name not in TOOL_SPECS:
        return []
    return [
        field for field in TOOL_SPECS[tool_name].required if not arguments.get(field)
    ]


def build_clarification_plan(
    request: ChatRequest,
    intent: Intent,
) -> ClarificationPlan:
    """Build the authoritative plan from explicit text and trusted context."""

    tool_name = select_realtime_tool(intent, request.user_message)
    arguments = _known_arguments(request, intent, tool_name)
    missing = _missing_required(tool_name, arguments)
    return ClarificationPlan(
        intent=intent,
        tool_name=tool_name,
        known_arguments=arguments,
        missing_required=missing,
        clarification_question=None,
        confidence=0.95,
        reason=(
            "后端依据结构化意图、显式业务编号和可信 Runtime Context 生成工具参数计划。"
        ),
        source="backend_guard",
    )


def apply_model_clarification_draft(
    authoritative_plan: ClarificationPlan,
    payload: dict[str, Any],
    *,
    model_name: str,
) -> ClarificationPlan:
    """Accept wording signals from the model but never its identity or arguments."""

    question = " ".join(
        str(payload.get("clarification_question") or "").split()
    ).strip()
    missing_field = (
        authoritative_plan.missing_required[0]
        if authoritative_plan.missing_required
        else None
    )
    related_terms = CLARIFICATION_FIELD_TERMS.get(str(missing_field), [])
    if (
        not question
        or len(question) > 120
        or not any(term.lower() in question.lower() for term in related_terms)
        or any(term.lower() in question.lower() for term in FORBIDDEN_CLARIFICATION_TERMS)
    ):
        question = None
    try:
        confidence = float(payload.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(confidence, 1.0))
    proposed_tool = str(payload.get("tool_name") or "").strip()
    tool_matches = proposed_tool in {"", str(authoritative_plan.tool_name or "")}
    reason = " ".join(
        str(payload.get("reason") or "模型生成澄清问题草案。").split()
    )[:160]
    if not tool_matches:
        reason = f"模型提议的工具与后端计划不一致，已忽略。{reason}"
    return authoritative_plan.model_copy(
        update={
            "clarification_question": question,
            "confidence": confidence,
            "reason": reason,
            "source": "model",
            "model_name": model_name,
        }
    )


def pre_tool_clarification(
    request: ChatRequest,
    plan: ClarificationPlan,
) -> ClarificationRequest | None:
    """Stop before tool execution whenever the backend still finds a missing field."""

    missing = _missing_required(plan.tool_name, plan.known_arguments)
    if not missing:
        return None
    field = missing[0]
    if field == "order_id":
        return ClarificationRequest(
            clarification_field=field,
            message=(
                plan.clarification_question
                or "你要查哪一个订单？请选择候选订单，或直接补充订单号。"
            ),
            candidates=order_candidates(request),
        )
    if field == "refund_request_id":
        return ClarificationRequest(
            clarification_field=field,
            message=(
                plan.clarification_question
                or "请提供退款申请号（例如 RF-1001），我才能查询退款进度。"
            ),
        )
    if field == "sku":
        return ClarificationRequest(
            clarification_field=field,
            message=(
                plan.clarification_question
                or "请补充商品名称或 SKU，我才能查询实时价格和库存。"
            ),
        )
    return ClarificationRequest(
        clarification_field=field,
        message=plan.clarification_question or f"请补充工具必填字段：{field}。",
    )


def post_tool_clarification(
    records: list[ToolCallRecord],
) -> ClarificationRequest | None:
    """Turn ambiguous observations into user choices instead of model guesses."""

    for record in records:
        observation = record.observation
        if observation.error_code == "product_ambiguous":
            candidates = [
                ClarificationCandidate(
                    value=str(item.get("code") or item.get("name") or ""),
                    label=str(item.get("name") or item.get("code") or "候选商品"),
                    hint=str(item.get("code") or "请补充准确商品名称"),
                )
                for item in observation.data.get("candidates", [])
                if isinstance(item, dict) and (item.get("code") or item.get("name"))
            ]
            return ClarificationRequest(
                clarification_field="sku",
                message="查到了多个候选商品，请确认你想查询哪一个 SKU。",
                candidates=candidates,
            )
        if record.action.tool_name == "search_current_user_orders":
            orders = observation.data.get("candidate_orders", [])
            if isinstance(orders, list) and len(orders) > 1:
                candidates = [
                    ClarificationCandidate(
                        value=str(item.get("value") or item.get("order_id") or ""),
                        label=str(
                            item.get("label")
                            or f"{item.get('order_id')}｜{item.get('status') or '状态待查'}"
                        ),
                        hint=str(
                            item.get("hint")
                            or "、".join(item.get("items") or [])
                            or "当前用户订单"
                        ),
                    )
                    for item in orders
                    if isinstance(item, dict)
                    and (item.get("value") or item.get("order_id"))
                ]
                return ClarificationRequest(
                    clarification_field="order_id",
                    message="按条件查到了多笔候选订单，请确认要查询哪一个订单号。",
                    candidates=candidates,
                )
    return None


def should_route_to_realtime_tool(intent: Intent, message: str) -> bool:
    """Keep stable policies in RAG and route changing facts to tools."""

    if intent == "refund_status_query":
        return True
    if intent == "order_query":
        return is_realtime_business_query(message) or bool(extract_order_id(message))
    if intent != "product_consult":
        return False
    realtime_terms = PRODUCT_REALTIME_TERMS
    stable_terms = ["规则", "活动规则", "优惠券", "叠加", "价保规则"]
    return any(term in message for term in realtime_terms) and not any(
        term in message for term in stable_terms
    )
