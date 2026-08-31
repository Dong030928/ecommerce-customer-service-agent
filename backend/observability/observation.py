"""Compress internal ToolResult payloads into model-safe Observations."""

from __future__ import annotations

from typing import Any

from api.schemas import ToolObservation, ToolResult


ERROR_SUMMARIES = {
    "tool_not_allowed": "模型提出了未开放的工具。",
    "tool_arguments_missing": "工具缺少必填参数。",
    "tool_arguments_not_allowed": "工具参数包含未允许字段，已拒绝执行。",
    "order_id_invalid": "订单号格式无效。",
    "refund_request_id_invalid": "退款申请号格式无效。",
    "order_month_invalid": "订单月份必须是 1 到 12 的整数。",
    "business_access_denied": "业务系统拒绝了本次查询，请确认记录归属和服务身份。",
    "business_fact_not_found": "业务系统没有查到对应记录。",
    "business_service_unavailable": "业务系统暂时不可用，请稍后重试。",
    "business_request_failed": "业务事实查询失败，请核对参数后重试。",
    "business_response_invalid": "业务系统没有返回可信的成功结果。",
    "runtime_identity_missing": "缺少可信用户身份，不能查询用户业务事实。",
    "service_auth_missing": "电商后端服务凭证未配置，暂时不能查询实时业务事实。",
    "product_not_found": "没有查到这个商品。",
    "product_ambiguous": "没有唯一匹配的商品，请确认准确 SKU。",
    "tool_not_executed": "工具未执行。",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _omitted_paths(
    prefix: str,
    payload: dict[str, Any],
    allowed: set[str],
) -> list[str]:
    return sorted(f"{prefix}.{key}" for key in payload if key not in allowed)


def _error_observation(result: ToolResult) -> ToolObservation:
    payload = result.raw_payload
    error_code = result.error_code or str(payload.get("error_code") or "tool_error")
    if error_code == "product_ambiguous":
        raw_candidates = payload.get("candidates")
        candidates = [
            {
                "code": item.get("code"),
                "name": item.get("name"),
            }
            for item in raw_candidates or []
            if isinstance(item, dict) and (item.get("code") or item.get("name"))
        ][:5]
        return ToolObservation(
            tool_name=result.tool_name,
            status="error",
            summary=ERROR_SUMMARIES[error_code],
            facts={"candidate_count": len(candidates)},
            omitted_fields=["candidates.* except code/name"],
            next_action="ask_clarification",
            data={"candidates": candidates},
            error_code=error_code,
            source=result.source,
        )

    safe_data = {
        key: payload[key]
        for key in ("missing", "unexpected")
        if isinstance(payload.get(key), list)
    }
    next_action = (
        "ask_clarification"
        if error_code == "tool_arguments_missing"
        else "fallback_answer"
    )
    return ToolObservation(
        tool_name=result.tool_name,
        status="error",
        summary=str(
            payload.get("safe_message")
            or ERROR_SUMMARIES.get(error_code)
            or "工具返回错误。"
        ),
        facts={"error": error_code},
        omitted_fields=sorted(
            key
            for key in payload
            if key not in {"error_code", "safe_message", "missing", "unexpected"}
        ),
        next_action=next_action,
        data=safe_data,
        error_code=error_code,
        source=result.source,
    )


def _order_observation(result: ToolResult) -> ToolObservation:
    order = _dict(result.raw_payload.get("order"))
    order_id = order.get("orderNo")
    status = order.get("status")
    payment_status = order.get("paymentStatus")
    allowed = {"orderNo", "status", "paymentStatus"}
    facts = {
        "order_id": order_id,
        "order_status": status,
        "payment_status": payment_status,
    }
    return ToolObservation(
        tool_name=result.tool_name,
        status="success",
        summary=(
            f"订单 {order_id} 当前状态为 {status or '待查'}，支付状态为 "
            f"{payment_status or '待查'}。"
        ),
        facts=facts,
        omitted_fields=_omitted_paths("order", order, allowed),
        next_action="answer_user",
        data={"order": facts},
        source=result.source,
    )


def _logistics_observation(result: ToolResult) -> ToolObservation:
    order = _dict(result.raw_payload.get("order"))
    logistics = _dict(result.raw_payload.get("logistics"))
    order_id = order.get("orderNo")
    latest_event = logistics.get("latestUpdate")
    if not latest_event:
        events = logistics.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            latest_event = events[0].get("content")
    facts = {
        "order_id": order_id,
        "order_status": order.get("status"),
        "logistics_status": logistics.get("status"),
        "carrier": logistics.get("company"),
        "latest_event": latest_event,
        "estimated_delivery": logistics.get("estimatedDelivery"),
    }
    if logistics:
        summary = (
            f"订单 {order_id} 的物流状态为 {facts['logistics_status'] or '待查'}，"
            f"承运方为 {facts['carrier'] or '待查'}，最新轨迹："
            f"{latest_event or '暂无'}。"
        )
    else:
        summary = (
            f"订单 {order_id} 当前状态为 {order.get('status') or '待查'}，"
            "暂未查到独立物流轨迹。"
        )
    omitted = _omitted_paths(
        "order",
        order,
        {"orderNo", "status"},
    ) + _omitted_paths(
        "logistics",
        logistics,
        {"status", "company", "latestUpdate", "estimatedDelivery"},
    )
    return ToolObservation(
        tool_name=result.tool_name,
        status="success",
        summary=summary,
        facts=facts,
        omitted_fields=sorted(omitted),
        next_action="answer_user",
        data={"logistics": facts},
        source=result.source,
    )


def _product_observation(result: ToolResult) -> ToolObservation:
    product = _dict(result.raw_payload.get("product"))
    facts = {
        "sku": product.get("code"),
        "name": product.get("name"),
        "current_price": product.get("price"),
        "inventory": product.get("stock"),
        "active": product.get("active"),
    }
    return ToolObservation(
        tool_name=result.tool_name,
        status="success",
        summary=(
            f"{facts['name']}（{facts['sku']}）当前价 {facts['current_price']} 元，"
            f"库存 {facts['inventory']} 件。"
        ),
        facts=facts,
        omitted_fields=_omitted_paths(
            "product",
            product,
            {"code", "name", "price", "stock", "active"},
        ),
        next_action="answer_user",
        data={"product": facts},
        source=result.source,
    )


def _refund_observation(result: ToolResult) -> ToolObservation:
    refund = _dict(result.raw_payload.get("refund"))
    facts = {
        "refund_request_id": refund.get("requestId"),
        "order_id": refund.get("orderNo"),
        "refund_status": refund.get("status"),
        "amount": refund.get("amount"),
        "updated_at": refund.get("updatedAt"),
    }
    return ToolObservation(
        tool_name=result.tool_name,
        status="success",
        summary=(
            f"退款申请 {facts['refund_request_id']} 当前状态为 "
            f"{facts['refund_status'] or '待查'}。"
        ),
        facts=facts,
        omitted_fields=_omitted_paths(
            "refund",
            refund,
            {"requestId", "orderNo", "status", "amount", "updatedAt"},
        ),
        next_action="answer_user",
        data={"refund": facts},
        source=result.source,
    )


def _candidate_orders_observation(result: ToolResult) -> ToolObservation:
    payload = result.raw_payload
    candidates = [
        item
        for item in payload.get("candidate_orders", [])
        if isinstance(item, dict)
    ]
    count = len(candidates)
    next_action = "ask_clarification" if count > 1 else "answer_user"
    return ToolObservation(
        tool_name=result.tool_name,
        status="success",
        summary=f"按 {payload.get('month')} 月找到 {count} 个当前用户订单候选。",
        facts={
            "month": payload.get("month"),
            "candidate_count": count,
            "context_truncated": payload.get("context_truncated") is True,
        },
        omitted_fields=["currentUserOrders.* except candidate label/hint"],
        next_action=next_action,
        data={"candidate_orders": candidates},
        source=result.source,
    )


def build_observation(result: ToolResult) -> ToolObservation:
    """Return the only representation allowed into model and public contexts."""

    if result.status != "success":
        return _error_observation(result)
    builders = {
        "get_order_status": _order_observation,
        "get_order_logistics": _logistics_observation,
        "get_product_inventory": _product_observation,
        "get_refund_status": _refund_observation,
        "search_current_user_orders": _candidate_orders_observation,
    }
    builder = builders.get(result.tool_name)
    if builder is None:
        return ToolObservation(
            tool_name=result.tool_name,
            status="error",
            summary="工具结果没有可用的安全压缩规则。",
            facts={"error": "observation_rule_missing"},
            omitted_fields=sorted(result.raw_payload),
            next_action="fallback_answer",
            error_code="observation_rule_missing",
            source="observation_builder",
        )
    return builder(result)
