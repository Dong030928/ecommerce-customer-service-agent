"""Offline tests for MCP-style Tool, Resource, and Prompt organization."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from agents.customer_service_agent import CustomerServiceAgent  # noqa: E402
from api.schemas import (  # noqa: E402
    ChatRequest,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from mcp_catalog.catalog import MCP_CATALOG  # noqa: E402
from tools.contracts import TOOL_SPECS  # noqa: E402
from tools.tool_calling import ToolCallingOutcome  # noqa: E402


class CatalogToolService:
    """Return one deterministic record so Agent binding output can be tested."""

    def run(self, *args: object, **kwargs: object) -> ToolCallingOutcome:
        record = ToolCallRecord(
            action=ToolAction(
                tool_name="get_order_status",
                arguments={"order_id": "SO20260420103000001-a1000001"},
                reason="从 MCP-style 目录选择只读订单工具。",
            ),
            observation=ToolObservation(
                tool_name="get_order_status",
                status="success",
                summary="订单当前状态为 SHIPPED。",
                facts={"order_status": "SHIPPED"},
                data={"order": {"order_status": "SHIPPED"}},
            ),
        )
        return ToolCallingOutcome(
            answer="订单当前状态为 SHIPPED。",
            tool_calls=[record],
            state={
                "create_agent": True,
                "tool_source": "mcp_catalog",
                "answer_source": "compressed_observation_fallback",
            },
            used_model=True,
            model_name="catalog-test-model",
        )


class MCPCatalogTests(unittest.TestCase):
    def _request(self, message: str) -> ChatRequest:
        return ChatRequest(
            session_id="mcp-catalog-test",
            runtime_user_id="PRIVATE-RUNTIME-USER",
            user_message=message,
        )

    def test_catalog_is_the_single_source_for_all_five_tool_specs(self) -> None:
        catalog_specs = MCP_CATALOG.to_tool_specs()

        self.assertEqual(set(catalog_specs), set(TOOL_SPECS))
        self.assertEqual(len(catalog_specs), 5)
        self.assertEqual(
            {
                name: spec.model_dump()
                for name, spec in catalog_specs.items()
            },
            {
                name: spec.model_dump()
                for name, spec in TOOL_SPECS.items()
            },
        )

    def test_every_tool_binding_resolves_to_resources_and_prompts(self) -> None:
        for definition in MCP_CATALOG.list_tools():
            self.assertTrue(definition.resource_uris)
            self.assertTrue(definition.prompt_ids)
            for uri in definition.resource_uris:
                self.assertEqual(MCP_CATALOG.read_resource(uri).uri, uri)
            for prompt_id in definition.prompt_ids:
                self.assertEqual(
                    MCP_CATALOG.get_prompt(prompt_id).prompt_id,
                    prompt_id,
                )

    def test_tool_route_exposes_selected_catalog_bindings(self) -> None:
        response = CustomerServiceAgent(
            tool_calling_service=CatalogToolService()
        ).chat(
            self._request(
                "查询订单 SO20260420103000001-a1000001 的状态"
            )
        )

        self.assertEqual(response.mcp_context.tool_source, "mcp_catalog")
        self.assertEqual(
            response.mcp_context.selected_tools,
            ["get_order_status"],
        )
        self.assertIn(
            "resource://ecommerce/tools/order-status-boundary",
            response.mcp_context.resources,
        )
        self.assertIn(
            "prompt://ecommerce/tool-observation",
            response.mcp_context.prompts,
        )
        self.assertFalse(response.mcp_context.remote_server_connected)
        self.assertEqual(
            response.session_state["mcp"],
            response.mcp_context.model_dump(),
        )

    def test_non_tool_route_reports_no_selected_catalog_tool(self) -> None:
        response = CustomerServiceAgent().chat(self._request("你好"))

        self.assertIsNone(response.mcp_context.selected_tool)
        self.assertEqual(response.mcp_context.selected_tools, [])
        self.assertEqual(response.mcp_context.resources, [])
        self.assertEqual(response.mcp_context.prompts, [])

    def test_high_risk_route_uses_boundary_binding_without_approval(self) -> None:
        response = CustomerServiceAgent().chat(
            self._request("请直接退款并赔付")
        )

        self.assertIn(
            "resource://ecommerce/high-risk-boundary",
            response.mcp_context.resources,
        )
        self.assertIn(
            "prompt://ecommerce/handoff-boundary",
            response.mcp_context.prompts,
        )
        self.assertEqual(response.mcp_context.selected_tools, [])
        self.assertTrue(response.needs_human_approval)
        self.assertFalse(
            response.session_state["hooks"]["hitl_approval_performed"]
        )


if __name__ == "__main__":
    unittest.main()
