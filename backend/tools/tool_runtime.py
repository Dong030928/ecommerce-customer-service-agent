"""Backend-controlled execution for allow-listed read-only tools."""

from __future__ import annotations

from typing import Any

from api.schemas import ChatRequest, ToolAction, ToolObservation
from integrations.ecommerce_client import EcommerceClient, EcommerceClientError
from tools.contracts import TOOL_SPECS
from tools.planning import ORDER_ID_PATTERN, REFUND_ID_PATTERN
from tools.runtime_context import (
    current_user_orders_truncated,
    order_candidates,
)


def _pick(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: data.get(field) for field in fields if field in data}


def sanitize_order(order: dict[str, Any]) -> dict[str, Any]:
    """Remove user identity and customer fields before model/public observation."""

    sanitized = _pick(
        order,
        [
            "orderNo",
            "status",
            "paymentStatus",
            "totalAmount",
            "createdAt",
            "shippedAt",
            "deliveredAt",
            "hasAfterSaleRequest",
            "cancelAllowed",
            "items",
        ],
    )
    return sanitized


def sanitize_logistics(logistics: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        logistics,
        [
            "company",
            "trackingNo",
            "status",
            "estimatedDelivery",
            "latestUpdate",
            "deliveredAt",
            "exceptionReason",
            "events",
        ],
    )


def sanitize_product(product: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        product,
        [
            "id",
            "code",
            "name",
            "category",
            "description",
            "price",
            "stock",
            "highlights",
            "active",
            "returnable",
            "afterSaleLimit",
            "scenarioTags",
            "promotion",
        ],
    )


def sanitize_refund(refund: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        refund,
        [
            "requestId",
            "orderNo",
            "amount",
            "reason",
            "status",
            "approvalId",
            "createdAt",
            "updatedAt",
        ],
    )


def validate_tool_action(action: ToolAction) -> ToolObservation | None:
    """Reject unknown tools, missing fields, and model-invented identity fields."""

    spec = TOOL_SPECS.get(action.tool_name)
    if spec is None:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary="模型提出了未开放的工具。",
            error_code="tool_not_allowed",
            source="tool_runtime",
        )
    missing = [field for field in spec.required if not action.arguments.get(field)]
    if missing:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary=f"工具参数缺失：{', '.join(missing)}。",
            data={"missing": missing},
            error_code="tool_arguments_missing",
            source="tool_runtime",
        )
    unexpected = sorted(set(action.arguments) - set(spec.parameters_schema))
    if unexpected:
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary="工具参数包含未允许字段，已拒绝执行。",
            data={"unexpected": unexpected},
            error_code="tool_arguments_not_allowed",
            source="tool_runtime",
        )
    if action.tool_name in {"get_order_status", "get_order_logistics"}:
        order_id = str(action.arguments["order_id"])
        if ORDER_ID_PATTERN.fullmatch(order_id) is None:
            return ToolObservation(
                tool_name=action.tool_name,
                status="error",
                summary="订单号格式无效。",
                error_code="order_id_invalid",
                source="tool_runtime",
            )
    if action.tool_name == "get_refund_status":
        refund_id = str(action.arguments["refund_request_id"])
        if REFUND_ID_PATTERN.fullmatch(refund_id) is None:
            return ToolObservation(
                tool_name=action.tool_name,
                status="error",
                summary="退款申请号格式无效。",
                error_code="refund_request_id_invalid",
                source="tool_runtime",
            )
    if action.tool_name == "search_current_user_orders":
        month = action.arguments.get("month")
        if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
            return ToolObservation(
                tool_name=action.tool_name,
                status="error",
                summary="订单月份必须是 1 到 12 的整数。",
                error_code="order_month_invalid",
                source="tool_runtime",
            )
    return None


class ToolRuntime:
    """Inject trusted identity and execute business reads after validation."""

    def __init__(self, ecommerce_client: EcommerceClient | None = None) -> None:
        self._ecommerce_client = ecommerce_client or EcommerceClient()

    def execute(self, action: ToolAction, request: ChatRequest) -> ToolObservation:
        validation_error = validate_tool_action(action)
        if validation_error is not None:
            return validation_error
        try:
            if action.tool_name == "search_current_user_orders":
                month = int(action.arguments["month"])
                candidates = order_candidates(request, month=month)
                truncated = current_user_orders_truncated(request)
                summary = f"按 {month} 月找到 {len(candidates)} 个当前用户订单候选。"
                if truncated:
                    summary += " 当前订单上下文已截断，结果只代表已加载窗口。"
                return ToolObservation(
                    tool_name=action.tool_name,
                    status="success",
                    summary=summary,
                    data={
                        "month": month,
                        "context_truncated": truncated,
                        "candidate_orders": [
                            candidate.model_dump() for candidate in candidates
                        ],
                    },
                    source="trusted_runtime_context",
                )
            if action.tool_name == "get_order_status":
                order = self._ecommerce_client.get_order(
                    str(action.arguments["order_id"]),
                    request.runtime_user_id,
                )
                safe_order = sanitize_order(order)
                return ToolObservation(
                    tool_name=action.tool_name,
                    status="success",
                    summary=(
                        f"订单 {safe_order.get('orderNo')} 当前状态为 "
                        f"{safe_order.get('status') or '待查'}，支付状态为 "
                        f"{safe_order.get('paymentStatus') or '待查'}。"
                    ),
                    data={"order": safe_order},
                )
            if action.tool_name == "get_order_logistics":
                order_id = str(action.arguments["order_id"])
                order = self._ecommerce_client.get_order(
                    order_id,
                    request.runtime_user_id,
                )
                logistics = self._ecommerce_client.get_logistics(
                    order_id,
                    request.runtime_user_id,
                )
                safe_order = sanitize_order(order)
                safe_logistics = sanitize_logistics(logistics)
                return ToolObservation(
                    tool_name=action.tool_name,
                    status="success",
                    summary=(
                        f"订单 {safe_order.get('orderNo')} 的物流状态为 "
                        f"{safe_logistics.get('status') or '待查'}，最新轨迹："
                        f"{safe_logistics.get('latestUpdate') or '暂无'}。"
                    ),
                    data={"order": safe_order, "logistics": safe_logistics},
                )
            if action.tool_name == "get_product_inventory":
                query = str(action.arguments["sku"]).strip()
                products = self._ecommerce_client.list_products(query)
                exact = [
                    product
                    for product in products
                    if str(product.get("code") or "").lower() == query.lower()
                    or str(product.get("name") or "").lower() == query.lower()
                ]
                candidates = exact or products
                if len(candidates) != 1:
                    public_candidates = [
                        _pick(product, ["code", "name"]) for product in candidates[:5]
                    ]
                    return ToolObservation(
                        tool_name=action.tool_name,
                        status="error",
                        summary="没有唯一匹配的商品，请补充准确 SKU 或商品名称。",
                        data={"candidates": public_candidates},
                        error_code="product_ambiguous",
                    )
                safe_product = sanitize_product(candidates[0])
                return ToolObservation(
                    tool_name=action.tool_name,
                    status="success",
                    summary=(
                        f"{safe_product.get('name')}（{safe_product.get('code')}）"
                        f"当前价 {safe_product.get('price')} 元，库存 "
                        f"{safe_product.get('stock')} 件。"
                    ),
                    data={"product": safe_product},
                )
            if action.tool_name == "get_refund_status":
                refund = self._ecommerce_client.get_refund_status(
                    str(action.arguments["refund_request_id"]),
                    request.runtime_user_id,
                )
                safe_refund = sanitize_refund(refund)
                return ToolObservation(
                    tool_name=action.tool_name,
                    status="success",
                    summary=(
                        f"退款申请 {safe_refund.get('requestId')} 当前状态为 "
                        f"{safe_refund.get('status') or '待查'}。"
                    ),
                    data={"refund": safe_refund},
                )
        except EcommerceClientError as exc:
            return ToolObservation(
                tool_name=action.tool_name,
                status="error",
                summary=exc.safe_message,
                error_code=exc.code,
            )
        return ToolObservation(
            tool_name=action.tool_name,
            status="error",
            summary="工具未执行。",
            error_code="tool_not_executed",
            source="tool_runtime",
        )
