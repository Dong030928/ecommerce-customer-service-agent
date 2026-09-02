"""TaskPlanner that emits a bounded and allow-listed RoutePlan."""

from __future__ import annotations

from typing import Any

from api.schemas import (
    ChatRequest,
    ExecutionRoute,
    Intent,
    IntentResult,
    PlannerTrace,
    RiskLevel,
    RoutePlan,
    ToolCandidate,
)
from degradation.fallbacks import is_high_risk_write_request
from mcp_catalog.catalog import MCP_CATALOG
from models.task_planner_client import TaskPlannerModelClient
from tools.planning import (
    extract_month,
    extract_order_id,
    extract_refund_request_id,
    extract_sku,
    is_product_tool_rag_query,
    select_realtime_tool,
    should_route_to_realtime_tool,
)


TOOL_DOMAINS = {
    "get_order_status": "order",
    "get_order_logistics": "order",
    "get_product_inventory": "product",
    "get_refund_status": "after_sale",
    "search_current_user_orders": "order",
}
KNOWLEDGE_DOMAINS = {
    "promotion_consult": ["promotion", "member"],
    "product_consult": ["product", "promotion"],
    "refund_request": ["after_sale"],
    "refund_status_query": ["after_sale"],
    "order_query": ["order", "shipping"],
    "complaint": ["complaint", "after_sale"],
}
ALLOWED_ENTITY_REFS = {"order_id", "refund_request_id", "sku", "month"}
ALLOWED_REQUIRED_CONTEXT = {"runtime_context"}
ALLOWED_FALLBACK_POLICIES = {
    "default",
    "knowledge_only",
    "tool_first",
    "tool_first_then_policy_caveat",
    "workflow_first",
    "clarify_or_model_plan",
}
MIN_MODEL_CONFIDENCE = 0.65
RULE_CONFIDENCE_THRESHOLD = 0.85


class TaskPlanner:
    """Choose one validated entry route without executing downstream work."""

    def __init__(
        self,
        model_client: TaskPlannerModelClient | None = None,
    ) -> None:
        self._model_client = model_client

    def plan(
        self,
        request: ChatRequest,
        rule_result: IntentResult | None,
    ) -> tuple[RoutePlan, PlannerTrace]:
        rule_plan = self._plan_by_rules(request.user_message, rule_result)
        safe_rule_plan, initial_override = self._apply_safety_constraints(
            request.user_message,
            rule_plan,
        )
        candidates = self._candidate_summaries(safe_rule_plan)
        if safe_rule_plan.confidence >= RULE_CONFIDENCE_THRESHOLD:
            constrained = self._constrain_required_tools(
                safe_rule_plan,
                candidates,
                request.user_message,
            )
            return constrained, self._trace(
                rule_plan,
                constrained,
                candidates,
                model_consulted=False,
                safety_override=initial_override,
            )

        payload = (
            self._model_client.plan(request.user_message, candidates)
            if self._model_client is not None
            else None
        )
        if payload is not None:
            model_plan = self._plan_from_model(
                payload,
                safe_rule_plan,
                candidates,
                request.user_message,
            )
            if model_plan.confidence >= MIN_MODEL_CONFIDENCE:
                safe_model_plan, model_override = self._apply_safety_constraints(
                    request.user_message,
                    model_plan,
                )
                constrained = self._constrain_required_tools(
                    safe_model_plan,
                    candidates,
                    request.user_message,
                )
                return constrained, self._trace(
                    rule_plan,
                    constrained,
                    candidates,
                    model_consulted=True,
                    safety_override=initial_override or model_override,
                )

        fallback = safe_rule_plan.model_copy(update={"source": "rules_fallback"})
        constrained = self._constrain_required_tools(
            fallback,
            candidates,
            request.user_message,
        )
        return constrained, self._trace(
            rule_plan,
            constrained,
            candidates,
            model_consulted=payload is not None,
            safety_override=initial_override,
        )

    def _plan_by_rules(
        self,
        message: str,
        result: IntentResult | None,
    ) -> RoutePlan:
        intent: Intent = result.intent if result is not None else "unknown"
        confidence = result.confidence if result is not None else 0.3
        source = result.source if result is not None else "rules_fallback"
        risk_level: RiskLevel = (
            "medium"
            if intent in {"refund_request", "refund_status_query", "complaint"}
            else "low"
        )
        if is_high_risk_write_request(message):
            return self._route_plan(
                intent=intent,
                execution_route="workflow",
                needs_rag=intent in {"refund_request", "complaint"},
                needs_business_tools=False,
                rag_query=message,
                confidence=max(confidence, 0.95),
                source="rules",
                required_tools=[],
                knowledge_domains=KNOWLEDGE_DOMAINS.get(intent, []),
                risk_level="high",
                requires_workflow=True,
                fallback_policy="workflow_first",
                entity_refs=self._entity_refs(message),
            )

        if is_product_tool_rag_query(intent, message):
            return self._route_plan(
                intent=intent,
                execution_route="tool_rag",
                needs_rag=True,
                needs_business_tools=True,
                rag_query=message,
                confidence=max(confidence, 0.95),
                source=source,
                required_tools=["get_product_inventory"],
                knowledge_domains=KNOWLEDGE_DOMAINS[intent],
                has_realtime_fact=True,
                required_context=["runtime_context"],
                fallback_policy="tool_first_then_policy_caveat",
                entity_refs=self._entity_refs(message),
            )

        if should_route_to_realtime_tool(intent, message):
            tool_name = select_realtime_tool(intent, message)
            return self._route_plan(
                intent=intent,
                execution_route="tool",
                needs_rag=False,
                needs_business_tools=tool_name is not None,
                rag_query=message,
                confidence=confidence,
                source=source,
                required_tools=[tool_name] if tool_name else [],
                knowledge_domains=[],
                has_realtime_fact=True,
                required_context=["runtime_context"],
                risk_level=risk_level,
                fallback_policy="tool_first",
                entity_refs=self._entity_refs(message),
            )

        needs_rag = intent not in {"general_chat", "unknown"}
        return self._route_plan(
            intent=intent,
            execution_route="rag" if needs_rag else "general",
            needs_rag=needs_rag,
            needs_business_tools=False,
            rag_query=message,
            confidence=confidence,
            source=source,
            required_tools=[],
            knowledge_domains=KNOWLEDGE_DOMAINS.get(intent, []),
            risk_level=risk_level,
            fallback_policy="knowledge_only" if needs_rag else "default",
            entity_refs=self._entity_refs(message),
        )

    def _plan_from_model(
        self,
        payload: dict[str, Any],
        fallback: RoutePlan,
        candidates: list[ToolCandidate],
        message: str,
    ) -> RoutePlan:
        allowed_intents = set(Intent.__args__)  # type: ignore[attr-defined]
        proposed_intent = str(payload.get("intent") or fallback.intent)
        intent: Intent = (
            proposed_intent if proposed_intent in allowed_intents else fallback.intent
        )  # type: ignore[assignment]
        required_tools = self._allowed_tools(
            self._list_value(payload.get("required_tools")),
            candidates,
        )
        has_realtime_fact = self._bool_value(
            payload.get("has_realtime_fact"),
            fallback.has_realtime_fact,
        )
        needs_tools = self._bool_value(
            payload.get("needs_business_tools"),
            fallback.needs_business_tools,
        ) or has_realtime_fact
        if needs_tools and not required_tools:
            selected = select_realtime_tool(intent, message)
            required_tools = self._allowed_tools(
                [selected] if selected else [],
                candidates,
            )
        needs_tools = bool(required_tools)
        needs_rag = self._bool_value(payload.get("needs_rag"), fallback.needs_rag)
        requires_workflow = self._bool_value(
            payload.get("requires_workflow"),
            fallback.requires_workflow,
        )
        risk_level = str(payload.get("risk_level") or fallback.risk_level).lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = fallback.risk_level
        domains = [
            item
            for item in self._list_value(payload.get("knowledge_domains"))
            if item in {
                "promotion",
                "member",
                "after_sale",
                "shipping",
                "product",
                "order",
                "payment",
                "complaint",
            }
        ]
        fallback_policy = str(
            payload.get("fallback_policy") or fallback.fallback_policy
        )
        if fallback_policy not in ALLOWED_FALLBACK_POLICIES:
            fallback_policy = fallback.fallback_policy
        return self._route_plan(
            intent=intent,
            execution_route=self._execution_route(
                needs_rag,
                needs_tools,
                requires_workflow,
            ),
            needs_rag=needs_rag,
            needs_business_tools=needs_tools,
            rag_query=str(payload.get("rag_query") or fallback.rag_query or message)[
                :600
            ],
            confidence=self._confidence_value(
                payload.get("confidence"),
                fallback.confidence,
            ),
            source="classifier",
            intents=[
                item
                for item in self._list_value(payload.get("intents"))
                if item in allowed_intents
            ]
            or [intent],
            entity_refs=[
                item
                for item in self._list_value(payload.get("entity_refs"))
                if item in ALLOWED_ENTITY_REFS
            ]
            or fallback.entity_refs,
            required_context=[
                item
                for item in self._list_value(payload.get("required_context"))
                if item in ALLOWED_REQUIRED_CONTEXT
            ],
            required_tools=required_tools,
            knowledge_domains=domains or KNOWLEDGE_DOMAINS.get(intent, []),
            has_realtime_fact=has_realtime_fact,
            risk_level=risk_level,  # type: ignore[arg-type]
            requires_workflow=requires_workflow,
            fallback_policy=fallback_policy,
        )

    def _apply_safety_constraints(
        self,
        message: str,
        plan: RoutePlan,
    ) -> tuple[RoutePlan, bool]:
        high_risk = (
            is_high_risk_write_request(message)
            or plan.risk_level == "high"
            or plan.requires_workflow
        )
        if not high_risk:
            return plan, False
        updated = plan.model_copy(
            update={
                "execution_route": "workflow",
                "needs_business_tools": False,
                "required_tools": [],
                "risk_level": "high",
                "requires_workflow": True,
                "fallback_policy": "workflow_first",
            }
        )
        return updated, updated != plan

    def _constrain_required_tools(
        self,
        plan: RoutePlan,
        candidates: list[ToolCandidate],
        message: str,
    ) -> RoutePlan:
        if plan.requires_workflow:
            return plan.model_copy(
                update={
                    "execution_route": "workflow",
                    "needs_business_tools": False,
                    "required_tools": [],
                }
            )
        allowed = self._allowed_tools(plan.required_tools, candidates)
        if plan.needs_business_tools and not allowed:
            selected = select_realtime_tool(plan.intent, message)
            allowed = self._allowed_tools(
                [selected] if selected else [],
                candidates,
            )
        needs_tools = bool(allowed)
        return plan.model_copy(
            update={
                "execution_route": self._execution_route(
                    plan.needs_rag,
                    needs_tools,
                    False,
                ),
                "needs_business_tools": needs_tools,
                "required_tools": allowed,
            }
        )

    @staticmethod
    def _route_plan(
        *,
        intent: Intent,
        execution_route: ExecutionRoute,
        needs_rag: bool,
        needs_business_tools: bool,
        rag_query: str,
        confidence: float,
        source: str,
        required_tools: list[str],
        knowledge_domains: list[str],
        risk_level: RiskLevel = "low",
        requires_workflow: bool = False,
        has_realtime_fact: bool = False,
        fallback_policy: str = "default",
        intents: list[str] | None = None,
        entity_refs: list[str] | None = None,
        required_context: list[str] | None = None,
    ) -> RoutePlan:
        return RoutePlan(
            intent=intent,
            execution_route=execution_route,
            needs_rag=needs_rag,
            needs_business_tools=needs_business_tools,
            rag_query=rag_query,
            confidence=confidence,
            source=source,
            intents=intents or [intent],
            entity_refs=entity_refs or [],
            required_context=required_context or [],
            required_tools=required_tools,
            knowledge_domains=knowledge_domains,
            has_realtime_fact=has_realtime_fact,
            risk_level=risk_level,
            requires_workflow=requires_workflow,
            fallback_policy=fallback_policy,
        )

    @staticmethod
    def _execution_route(
        needs_rag: bool,
        needs_tools: bool,
        requires_workflow: bool,
    ) -> ExecutionRoute:
        if requires_workflow:
            return "workflow"
        if needs_rag and needs_tools:
            return "tool_rag"
        if needs_tools:
            return "tool"
        if needs_rag:
            return "rag"
        return "general"

    @staticmethod
    def _entity_refs(message: str) -> list[str]:
        refs: list[str] = []
        if extract_order_id(message):
            refs.append("order_id")
        if extract_refund_request_id(message):
            refs.append("refund_request_id")
        if extract_sku(message):
            refs.append("sku")
        if extract_month(message) is not None:
            refs.append("month")
        return refs

    @staticmethod
    def _candidate_summaries(plan: RoutePlan) -> list[ToolCandidate]:
        definitions = MCP_CATALOG.list_tools()
        if plan.requires_workflow:
            return []
        requested = set(plan.required_tools)
        candidates = [
            ToolCandidate(
                name=definition.name,
                domain=TOOL_DOMAINS[definition.name],
                allowed_in_light_path=definition.read_only,
                risk_level=definition.risk_level,
                reason=(
                    "MCP Catalog 中已注册的只读能力；最终参数仍由后端和 Hook 校验。"
                ),
            )
            for definition in definitions
            if not requested or definition.name in requested
        ]
        return candidates

    @staticmethod
    def _allowed_tools(
        required_tools: list[str],
        candidates: list[ToolCandidate],
    ) -> list[str]:
        allowed = {
            candidate.name
            for candidate in candidates
            if candidate.allowed_in_light_path
        }
        return list(dict.fromkeys(name for name in required_tools if name in allowed))

    @staticmethod
    def _list_value(value: Any) -> list[str]:
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _bool_value(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "是"}:
                return True
            if lowered in {"false", "0", "no", "否"}:
                return False
        return default

    @staticmethod
    def _confidence_value(value: Any, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _trace(
        rule_plan: RoutePlan,
        final_plan: RoutePlan,
        candidates: list[ToolCandidate],
        *,
        model_consulted: bool,
        safety_override: bool,
    ) -> PlannerTrace:
        reasons = {
            "workflow": (
                "本轮命中高风险写操作，只返回 Workflow 路由信号，"
                "不在轻路径执行退款、取消或赔付。"
            ),
            "tool_rag": "本轮同时需要稳定知识和实时事实，进入 Tool + RAG 路由。",
            "tool": "本轮需要实时业务事实，工具候选已按目录白名单收窄。",
            "rag": "本轮属于稳定知识咨询，进入 RAG 路由并要求可验证引用。",
            "general": "本轮没有命中知识、工具或受控流程，进入普通客服对话。",
        }
        return PlannerTrace(
            source=final_plan.source,
            rule_confidence=rule_plan.confidence,
            model_consulted=model_consulted,
            safety_override=safety_override,
            candidate_tools=candidates,
            constrained_required_tools=final_plan.required_tools,
            public_reason=reasons[final_plan.execution_route],
        )
