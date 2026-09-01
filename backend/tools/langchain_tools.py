"""Lazy LangChain StructuredTool adapters around the controlled runtime."""

from __future__ import annotations

from typing import Any

from api.schemas import ChatRequest, ToolAction
from hooks.manager import HookManager
from tools.contracts import TOOL_SPECS
from tools.tool_runtime import ToolRuntime


def tool_action_reason(tool_name: str) -> str:
    reasons = {
        "get_order_status": "用户询问实时订单状态，模型选择只读订单工具。",
        "get_order_logistics": "用户询问实时物流，模型选择只读物流工具。",
        "get_product_inventory": "用户询问实时价格或库存，模型选择只读商品工具。",
        "get_refund_status": "用户询问退款进度，模型选择只读退款状态工具。",
        "search_current_user_orders": "用户只提供了月份，模型先查询当前用户的候选订单。",
    }
    return reasons.get(tool_name, "模型根据工具描述提出只读工具调用。")


def build_langchain_tools(
    request: ChatRequest,
    runtime: ToolRuntime,
    hooks: HookManager | None = None,
) -> list[Any]:
    """Import LangChain only on the realtime route and keep execution controlled."""

    from langchain_core.tools import StructuredTool

    def run(tool_name: str, arguments: dict[str, Any]) -> str:
        observation = runtime.execute(
            ToolAction(
                tool_name=tool_name,
                arguments=arguments,
                reason=tool_action_reason(tool_name),
            ),
            request,
            hooks,
        )
        return observation.model_dump_json()

    def get_order_status(order_id: str) -> str:
        """查询当前用户订单状态。"""
        return run("get_order_status", {"order_id": order_id})

    def get_order_logistics(order_id: str) -> str:
        """查询当前用户订单物流。"""
        return run("get_order_logistics", {"order_id": order_id})

    def get_product_inventory(sku: str) -> str:
        """查询商品实时价格与库存。"""
        return run("get_product_inventory", {"sku": sku})

    def get_refund_status(refund_request_id: str) -> str:
        """查询当前用户退款申请状态。"""
        return run("get_refund_status", {"refund_request_id": refund_request_id})

    def search_current_user_orders(month: int) -> str:
        """按月份查询当前用户候选订单。"""
        return run("search_current_user_orders", {"month": month})

    functions = {
        "get_order_status": get_order_status,
        "get_order_logistics": get_order_logistics,
        "get_product_inventory": get_product_inventory,
        "get_refund_status": get_refund_status,
        "search_current_user_orders": search_current_user_orders,
    }
    return [
        StructuredTool.from_function(
            func=functions[name],
            name=spec.name,
            description=spec.description,
        )
        for name, spec in TOOL_SPECS.items()
    ]
