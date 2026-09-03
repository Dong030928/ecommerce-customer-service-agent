"""Standardized local source for Tool, Resource, and Prompt definitions."""

from __future__ import annotations

from api.schemas import (
    MCPBindingSummary,
    MCPPrompt,
    MCPResource,
    MCPToolDefinition,
    RiskLevel,
    ToolAction,
    ToolSpec,
)


class MCPCatalog:
    """Expose reusable MCP-style definitions without pretending to be a server."""

    def __init__(self) -> None:
        observation_prompt = "prompt://ecommerce/tool-observation"
        self._tools: dict[str, MCPToolDefinition] = {
            "get_order_status": MCPToolDefinition(
                name="get_order_status",
                description="查询当前登录用户某个订单的实时状态，只读。",
                required=["order_id"],
                parameters_schema={
                    "order_id": "订单号，例如 SO20260420103000001-a1000001"
                },
                read_only=True,
                risk_level="low",
                resource_uris=["resource://ecommerce/tools/order-status-boundary"],
                prompt_ids=[observation_prompt],
            ),
            "get_order_logistics": MCPToolDefinition(
                name="get_order_logistics",
                description="查询当前登录用户某个订单的实时物流轨迹，只读。",
                required=["order_id"],
                parameters_schema={
                    "order_id": "订单号，例如 SO20260420103000001-a1000001"
                },
                read_only=True,
                risk_level="low",
                resource_uris=["resource://ecommerce/tools/logistics-boundary"],
                prompt_ids=[observation_prompt],
            ),
            "get_product_inventory": MCPToolDefinition(
                name="get_product_inventory",
                description="按 SKU 或商品名称查询实时价格、库存和当前活动，只读。",
                required=["sku"],
                parameters_schema={
                    "sku": "商品 SKU 或明确商品名称，例如 SKU-AUD-101"
                },
                read_only=True,
                risk_level="low",
                resource_uris=["resource://ecommerce/tools/product-boundary"],
                prompt_ids=[observation_prompt],
            ),
            "get_refund_status": MCPToolDefinition(
                name="get_refund_status",
                description="按退款申请号查询当前登录用户的退款进度，只读，不创建退款。",
                required=["refund_request_id"],
                parameters_schema={
                    "refund_request_id": "退款申请号，例如 RF-1001"
                },
                read_only=True,
                risk_level="medium",
                resource_uris=["resource://ecommerce/tools/refund-status-boundary"],
                prompt_ids=[observation_prompt],
            ),
            "search_current_user_orders": MCPToolDefinition(
                name="search_current_user_orders",
                description="按月份筛选可信 Runtime Context 中当前用户的候选订单，只读。",
                required=["month"],
                parameters_schema={"month": "订单月份，整数 1 到 12"},
                read_only=True,
                risk_level="low",
                resource_uris=[
                    "resource://ecommerce/tools/order-candidates-boundary"
                ],
                prompt_ids=[observation_prompt],
            ),
        }
        self._resources: dict[str, MCPResource] = {
            "resource://ecommerce/tools/order-status-boundary": MCPResource(
                uri="resource://ecommerce/tools/order-status-boundary",
                title="订单状态工具边界",
                content=(
                    "订单状态工具只查询当前登录用户的订单事实，不能查询他人订单，"
                    "也不能取消或修改订单。"
                ),
            ),
            "resource://ecommerce/tools/logistics-boundary": MCPResource(
                uri="resource://ecommerce/tools/logistics-boundary",
                title="物流工具边界",
                content=(
                    "物流工具只返回当前登录用户订单的实时物流事实，不能编造包裹"
                    "位置，也不能查询他人订单。"
                ),
            ),
            "resource://ecommerce/tools/product-boundary": MCPResource(
                uri="resource://ecommerce/tools/product-boundary",
                title="商品工具边界",
                content=(
                    "商品工具只返回实时库存、价格和当前活动，不替代稳定商品知识库，"
                    "也不承诺最终结算价。"
                ),
            ),
            "resource://ecommerce/tools/refund-status-boundary": MCPResource(
                uri="resource://ecommerce/tools/refund-status-boundary",
                title="退款进度工具边界",
                content=(
                    "退款进度工具只查询已有申请状态，不创建退款、不取消订单、"
                    "不批准补偿。"
                ),
            ),
            "resource://ecommerce/tools/order-candidates-boundary": MCPResource(
                uri="resource://ecommerce/tools/order-candidates-boundary",
                title="候选订单工具边界",
                content=(
                    "候选订单只能来自可信 Runtime Context，并且必须由用户确认目标，"
                    "模型不能代替用户选择。"
                ),
            ),
            "resource://ecommerce/high-risk-boundary": MCPResource(
                uri="resource://ecommerce/high-risk-boundary",
                title="高风险动作边界",
                content=(
                    "退款、取消订单、补偿和改地址需要受控流程与人工确认，"
                    "不能由普通 Tool Use 自动执行。"
                ),
            ),
        }
        self._prompts: dict[str, MCPPrompt] = {
            observation_prompt: MCPPrompt(
                prompt_id=observation_prompt,
                title="工具 Observation 口径",
                content=(
                    "把工具返回压缩为事实摘要，保留必要字段和 omitted_fields，"
                    "不暴露原始 payload、凭证或内部调试字段。"
                ),
            ),
            "prompt://ecommerce/handoff-boundary": MCPPrompt(
                prompt_id="prompt://ecommerce/handoff-boundary",
                title="高风险转人工口径",
                content=(
                    "遇到退款、取消订单和补偿等高风险写动作时，不自动执行，"
                    "明确进入人工处理边界。"
                ),
            ),
        }
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        for definition in self._tools.values():
            missing_resources = set(definition.resource_uris) - set(self._resources)
            missing_prompts = set(definition.prompt_ids) - set(self._prompts)
            if missing_resources or missing_prompts:
                raise ValueError(
                    "MCP catalog contains unresolved bindings: "
                    f"resources={sorted(missing_resources)}, "
                    f"prompts={sorted(missing_prompts)}"
                )

    def list_tools(self) -> list[MCPToolDefinition]:
        """Return catalog definitions in deterministic name order."""

        return [self._tools[name] for name in sorted(self._tools)]

    def to_tool_specs(self) -> dict[str, ToolSpec]:
        """Adapt the catalog to the existing planner/runtime contracts."""

        return {
            name: self._tools[name].to_tool_spec()
            for name in sorted(self._tools)
        }

    def read_resource(self, uri: str) -> MCPResource:
        return self._resources[uri]

    def get_prompt(self, prompt_id: str) -> MCPPrompt:
        return self._prompts[prompt_id]

    def binding_summary(
        self,
        actions: list[ToolAction] | None,
        risk_level: RiskLevel,
    ) -> MCPBindingSummary:
        """Summarize bindings actually selected by Tool Use for this request."""

        available = sorted(self._tools)
        selected = sorted(
            {
                action.tool_name
                for action in actions or []
                if action.tool_name in self._tools
            }
        )
        if risk_level == "high":
            resources = sorted(
                {
                    "resource://ecommerce/high-risk-boundary",
                    *(
                        uri
                        for name in selected
                        for uri in self._tools[name].resource_uris
                    ),
                }
            )
            return MCPBindingSummary(
                selected_tool=selected[0] if selected else None,
                selected_tools=selected,
                available_tools=available,
                resources=resources,
                prompts=sorted(
                    {
                        "prompt://ecommerce/handoff-boundary",
                        *(
                            prompt_id
                            for name in selected
                            for prompt_id in self._tools[name].prompt_ids
                        ),
                    }
                ),
                boundary=(
                    "本地 MCP-style 目录只允许为高风险资格评估读取证据，"
                    "不执行写操作，也不替代 Workflow/HITL。"
                ),
            )
        resources = sorted(
            {
                uri
                for name in selected
                for uri in self._tools[name].resource_uris
            }
        )
        prompts = sorted(
            {
                prompt_id
                for name in selected
                for prompt_id in self._tools[name].prompt_ids
            }
        )
        boundary = (
            "本地 MCP-style 目录提供标准化工具、Resource 和 Prompt 绑定；"
            "Tool Use 仍负责规划与执行，Hooks 仍负责治理。"
            if selected
            else "本轮没有实际调用目录工具；Tool Use 仍负责决定是否调用。"
        )
        return MCPBindingSummary(
            selected_tool=selected[0] if selected else None,
            selected_tools=selected,
            available_tools=available,
            resources=resources,
            prompts=prompts,
            boundary=boundary,
        )


MCP_CATALOG = MCPCatalog()
