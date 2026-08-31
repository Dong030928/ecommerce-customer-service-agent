"""Allow-listed tool contracts visible to the planning model."""

from api.schemas import ToolSpec


TOOL_SPECS: dict[str, ToolSpec] = {
    "get_order_status": ToolSpec(
        name="get_order_status",
        description="查询当前登录用户某个订单的实时状态，只读。",
        required=["order_id"],
        parameters_schema={"order_id": "订单号，例如 SO20260420103000001-a1000001"},
    ),
    "get_order_logistics": ToolSpec(
        name="get_order_logistics",
        description="查询当前登录用户某个订单的实时物流轨迹，只读。",
        required=["order_id"],
        parameters_schema={"order_id": "订单号，例如 SO20260420103000001-a1000001"},
    ),
    "get_product_inventory": ToolSpec(
        name="get_product_inventory",
        description="按 SKU 或商品名称查询实时价格、库存和当前活动，只读。",
        required=["sku"],
        parameters_schema={"sku": "商品 SKU 或明确商品名称，例如 SKU-AUD-101"},
    ),
    "get_refund_status": ToolSpec(
        name="get_refund_status",
        description="按退款申请号查询当前登录用户的退款进度，只读，不创建退款。",
        required=["refund_request_id"],
        parameters_schema={"refund_request_id": "退款申请号，例如 RF-1001"},
    ),
    "search_current_user_orders": ToolSpec(
        name="search_current_user_orders",
        description="按月份筛选可信 Runtime Context 中当前用户的候选订单，只读。",
        required=["month"],
        parameters_schema={"month": "订单月份，整数 1 到 12"},
    ),
}
