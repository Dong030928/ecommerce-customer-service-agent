"""Minimize trusted runtime context before exposing local diagnostics."""

from __future__ import annotations

import re
from typing import Any

from api.schemas import ChatRequest, ClarificationCandidate


def runtime_context(request: ChatRequest) -> dict[str, Any]:
    return request.runtime_context if isinstance(request.runtime_context, dict) else {}


def _order_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _order_id(order: dict[str, Any]) -> str:
    return str(order.get("orderNo") or order.get("order_id") or "").strip()


def _order_status(order: dict[str, Any]) -> str:
    return str(order.get("status") or order.get("orderStatus") or "").strip()


def _order_items(order: dict[str, Any]) -> list[str]:
    items = order.get("itemSummary") or order.get("items") or []
    if isinstance(items, str):
        return [part.strip() for part in re.split(r"[，,、;；]", items) if part.strip()]
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("productName") or item.get("name")
            quantity = item.get("quantity")
            if name:
                names.append(f"{name} x{quantity}" if quantity else str(name))
    return names


def _created_month(order: dict[str, Any]) -> int | None:
    created_at = str(order.get("createdAt") or order.get("created_at") or "")
    match = re.search(r"^\d{4}-(\d{1,2})-", created_at)
    return int(match.group(1)) if match else None


def current_user_orders(request: ChatRequest) -> list[dict[str, Any]]:
    """Read order summaries supplied by the authenticated application gateway."""

    context = runtime_context(request)
    orders = _order_list(context.get("currentUserOrders"))
    related_order_id = str(context.get("relatedOrderNo") or "").strip()
    if related_order_id and all(_order_id(order) != related_order_id for order in orders):
        orders.append(
            {
                "orderNo": related_order_id,
                "status": "当前页面关联订单",
                "itemSummary": [],
            }
        )
    return orders


def current_user_orders_truncated(request: ChatRequest) -> bool:
    return runtime_context(request).get("currentUserOrdersTruncated") is True


def trusted_order_eligibility_facts(
    request: ChatRequest,
    order_id: str,
) -> dict[str, Any]:
    """Return a strict whitelist of gateway-provided facts for one exact order."""

    target = order_id.strip().lower()
    order = next(
        (
            item
            for item in current_user_orders(request)
            if _order_id(item).lower() == target
        ),
        None,
    )
    if order is None:
        return {}
    returnable = order.get("returnable")
    if not isinstance(returnable, bool):
        items = order.get("items")
        if isinstance(items, list) and items:
            flags = [
                item.get("returnable")
                for item in items
                if isinstance(item, dict)
            ]
            if len(flags) == len(items) and all(isinstance(flag, bool) for flag in flags):
                returnable = all(flags)
            else:
                returnable = None
        else:
            returnable = None
    return {
        "fulfillmentStatus": order.get("fulfillmentStatus"),
        "deliveredAt": order.get("deliveredAt"),
        "returnable": returnable,
    }


def contextual_order_id(request: ChatRequest) -> str | None:
    """Use a page-linked or unique trusted order, never guess among candidates."""

    related = str(runtime_context(request).get("relatedOrderNo") or "").strip()
    if related:
        return related
    orders = current_user_orders(request)
    if len(orders) == 1:
        return _order_id(orders[0]) or None
    return None


def order_candidates(
    request: ChatRequest,
    *,
    month: int | None = None,
) -> list[ClarificationCandidate]:
    orders = current_user_orders(request)
    if month is not None:
        orders = [order for order in orders if _created_month(order) == month]
    return [
        ClarificationCandidate(
            value=_order_id(order),
            label=f"{_order_id(order)}｜{_order_status(order) or '状态待查'}",
            hint="、".join(_order_items(order)) or "当前用户订单",
        )
        for order in orders
        if _order_id(order)
    ]


def public_runtime_context(request: ChatRequest) -> dict[str, Any]:
    """Expose identifiers useful to the caller, never embedded order payloads."""

    context = runtime_context(request)
    orders = current_user_orders(request)
    return {
        "current_page": context.get("currentPage"),
        "related_product_id": context.get("relatedProductId"),
        "related_order_no": context.get("relatedOrderNo"),
        "related_after_sale_no": context.get("relatedAfterSaleNo"),
        "current_user_order_count": len(orders),
        "current_user_orders_truncated": current_user_orders_truncated(request),
    }
