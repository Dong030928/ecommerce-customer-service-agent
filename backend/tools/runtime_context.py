"""Minimize trusted runtime context before exposing local diagnostics."""

from __future__ import annotations

from typing import Any

from api.schemas import ChatRequest


def runtime_context(request: ChatRequest) -> dict[str, Any]:
    return request.runtime_context if isinstance(request.runtime_context, dict) else {}


def public_runtime_context(request: ChatRequest) -> dict[str, Any]:
    """Expose identifiers useful to the caller, never embedded order payloads."""

    context = runtime_context(request)
    orders = context.get("currentUserOrders")
    return {
        "current_page": context.get("currentPage"),
        "related_product_id": context.get("relatedProductId"),
        "related_order_no": context.get("relatedOrderNo"),
        "related_after_sale_no": context.get("relatedAfterSaleNo"),
        "current_user_order_count": len(orders) if isinstance(orders, list) else 0,
    }
