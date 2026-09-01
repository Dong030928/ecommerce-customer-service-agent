"""Backend-controlled execution with internal ToolResult isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from api.schemas import ChatRequest, ToolAction, ToolObservation, ToolResult
from degradation.fallbacks import classify_error_code
from integrations.ecommerce_client import EcommerceClient, EcommerceClientError
from observability.observation import build_observation
from tools.contracts import TOOL_SPECS
from tools.planning import ORDER_ID_PATTERN, REFUND_ID_PATTERN
from tools.runtime_context import (
    current_user_orders_truncated,
    order_candidates,
)

if TYPE_CHECKING:
    from hooks.manager import HookManager


def _validation_result(action: ToolAction) -> ToolResult | None:
    spec = TOOL_SPECS.get(action.tool_name)
    if spec is None:
        return ToolResult(
            tool_name=action.tool_name,
            status="error",
            raw_payload={"error_code": "tool_not_allowed"},
            error_code="tool_not_allowed",
            error_category="validation_error",
            source="tool_runtime",
        )
    missing = [field for field in spec.required if not action.arguments.get(field)]
    if missing:
        return ToolResult(
            tool_name=action.tool_name,
            status="error",
            raw_payload={
                "error_code": "tool_arguments_missing",
                "missing": missing,
            },
            error_code="tool_arguments_missing",
            error_category="validation_error",
            source="tool_runtime",
        )
    unexpected = sorted(set(action.arguments) - set(spec.parameters_schema))
    if unexpected:
        return ToolResult(
            tool_name=action.tool_name,
            status="error",
            raw_payload={
                "error_code": "tool_arguments_not_allowed",
                "unexpected": unexpected,
            },
            error_code="tool_arguments_not_allowed",
            error_category="validation_error",
            source="tool_runtime",
        )
    if action.tool_name in {"get_order_status", "get_order_logistics"}:
        if ORDER_ID_PATTERN.fullmatch(str(action.arguments["order_id"])) is None:
            return ToolResult(
                tool_name=action.tool_name,
                status="error",
                raw_payload={"error_code": "order_id_invalid"},
                error_code="order_id_invalid",
                error_category="validation_error",
                source="tool_runtime",
            )
    if action.tool_name == "get_refund_status":
        if (
            REFUND_ID_PATTERN.fullmatch(
                str(action.arguments["refund_request_id"])
            )
            is None
        ):
            return ToolResult(
                tool_name=action.tool_name,
                status="error",
                raw_payload={"error_code": "refund_request_id_invalid"},
                error_code="refund_request_id_invalid",
                error_category="validation_error",
                source="tool_runtime",
            )
    if action.tool_name == "search_current_user_orders":
        month = action.arguments.get("month")
        if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
            return ToolResult(
                tool_name=action.tool_name,
                status="error",
                raw_payload={"error_code": "order_month_invalid"},
                error_code="order_month_invalid",
                error_category="validation_error",
                source="tool_runtime",
            )
    return None


def validate_tool_action(action: ToolAction) -> ToolObservation | None:
    """Keep a public-safe validation API while retaining raw results internally."""

    result = _validation_result(action)
    return build_observation(result) if result is not None else None


class ToolRuntime:
    """Execute raw business reads internally and expose only Observations."""

    def __init__(self, ecommerce_client: EcommerceClient | None = None) -> None:
        self._ecommerce_client = ecommerce_client or EcommerceClient()

    def _execute_once(self, action: ToolAction, request: ChatRequest) -> ToolResult:
        """Execute one attempt after validation; callers own retry policy."""

        if action.tool_name == "search_current_user_orders":
            month = int(action.arguments["month"])
            candidates = order_candidates(request, month=month)
            return ToolResult(
                tool_name=action.tool_name,
                status="success",
                raw_payload={
                    "month": month,
                    "context_truncated": current_user_orders_truncated(request),
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
            return ToolResult(
                tool_name=action.tool_name,
                status="success",
                raw_payload={"order": order},
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
            return ToolResult(
                tool_name=action.tool_name,
                status="success",
                raw_payload={"order": order, "logistics": logistics},
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
            if not candidates:
                return ToolResult(
                    tool_name=action.tool_name,
                    status="error",
                    raw_payload={"error_code": "product_not_found"},
                    error_code="product_not_found",
                    error_category="not_found",
                )
            if len(candidates) != 1:
                return ToolResult(
                    tool_name=action.tool_name,
                    status="error",
                    raw_payload={
                        "error_code": "product_ambiguous",
                        "candidates": candidates,
                    },
                    error_code="product_ambiguous",
                    error_category="validation_error",
                )
            return ToolResult(
                tool_name=action.tool_name,
                status="success",
                raw_payload={"product": candidates[0]},
            )
        if action.tool_name == "get_refund_status":
            refund = self._ecommerce_client.get_refund_status(
                str(action.arguments["refund_request_id"]),
                request.runtime_user_id,
            )
            return ToolResult(
                tool_name=action.tool_name,
                status="success",
                raw_payload={"refund": refund},
            )
        return ToolResult(
            tool_name=action.tool_name,
            status="error",
            raw_payload={"error_code": "tool_not_executed"},
            error_code="tool_not_executed",
            error_category="system_error",
            source="tool_runtime",
        )

    def execute_raw(self, action: ToolAction, request: ChatRequest) -> ToolResult:
        """Run a read with at most one retry, and only when it times out."""

        validation_error = _validation_result(action)
        if validation_error is not None:
            return validation_error
        spec = TOOL_SPECS[action.tool_name]
        max_attempts = 2 if spec.read_only else 1
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._execute_once(action, request)
                return result.model_copy(update={"attempts": attempt})
            except EcommerceClientError as exc:
                category = classify_error_code(exc.code)
                if category == "timeout" and attempt < max_attempts:
                    continue
                return ToolResult(
                    tool_name=action.tool_name,
                    status="error",
                    raw_payload={
                        "error_code": exc.code,
                        "safe_message": exc.safe_message,
                    },
                    error_code=exc.code,
                    error_category=category,
                    attempts=attempt,
                )
        raise AssertionError("tool retry loop exhausted without a result")

    def execute(
        self,
        action: ToolAction,
        request: ChatRequest,
        hooks: HookManager | None = None,
    ) -> ToolObservation:
        """Run lifecycle hooks around execution and expose only an Observation."""

        if hooks is not None:
            hooks.pre_tool_call(action, request, TOOL_SPECS.get(action.tool_name))
        observation = build_observation(self.execute_raw(action, request))
        if hooks is not None:
            observation = hooks.post_tool_call(observation)
            if observation.status == "error":
                hooks.on_error(
                    action.tool_name,
                    observation.error_category,
                    observation.summary,
                    observation.attempts,
                )
        return observation
