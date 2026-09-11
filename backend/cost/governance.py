"""Request-level path, cache, prompt, observation, and budget governance."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any

from config.settings import DEFAULT_REQUEST_TOKEN_BUDGET



@dataclass(frozen=True)
class CostGovernanceContext:
    """Public inputs used to explain cost without exposing raw model/tool payloads."""

    path_type: str
    intent: str
    tool_calls: list[Any] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    workflow: Any | None = None
    retrieval_cache: dict[str, Any] | None = None
    route_planner_model_used: bool = False
    final_answer_model_used: bool = False
    degraded: bool = False


def selected_prompt_fragments(context: CostGovernanceContext) -> list[str]:
    """Name the prompt policies actually needed by the selected execution path."""

    fragments = ["role_customer_service", "public_safety_boundary"]
    if context.path_type in {"tool_calling_path", "tool_rag_heavy_path"}:
        fragments.append("tool_observation_policy")
    if context.citations:
        fragments.append("rag_citation_answer_policy")
    if context.workflow is not None:
        fragments.append("high_risk_after_sale_hitl_boundary")
    if context.path_type == "security_guard_path":
        fragments.append("protected_information_refusal")
    return fragments


def _estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return max(1, ascii_chars // 4 + (len(text) - ascii_chars) // 2)


def observation_compression(tool_calls: list[Any], path_type: str) -> dict[str, Any]:
    """Measure model-visible selection from public Observations, never raw ToolResult."""

    public_tokens = 0
    model_context_tokens = 0
    for record in tool_calls:
        observation = getattr(record, "observation", None)
        if observation is None:
            continue
        payload = observation.model_dump(mode="json")
        public_tokens += _estimate_tokens(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if path_type == "tool_rag_heavy_path":
            selected = {
                key: payload.get(key)
                for key in ("tool_name", "status", "summary", "facts", "next_action")
            }
            model_context_tokens += _estimate_tokens(
                json.dumps(selected, ensure_ascii=False, sort_keys=True)
            )
        elif path_type == "tool_calling_path":
            # LangChain receives the complete sanitized Observation JSON.
            model_context_tokens += _estimate_tokens(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )
    return {
        "schema_version": "observation_compression_v1",
        "public_observation_tokens": public_tokens,
        "model_context_tokens": model_context_tokens,
        "saved_tokens": max(0, public_tokens - model_context_tokens),
        "strategy": (
            "joint_prompt_field_allowlist"
            if path_type == "tool_rag_heavy_path"
            else "sanitized_observation_passthrough"
            if path_type == "tool_calling_path"
            else "not_sent_to_answer_model"
        ),
        "upstream_tool_result_compressed": bool(tool_calls),
        "raw_tool_result_measured": False,
    }


def _token_budget() -> int:
    raw = os.getenv("AGENT_REQUEST_TOKEN_BUDGET")
    try:
        value = int(raw) if raw is not None else DEFAULT_REQUEST_TOKEN_BUDGET
    except ValueError:
        return DEFAULT_REQUEST_TOKEN_BUDGET
    return value if value > 0 else DEFAULT_REQUEST_TOKEN_BUDGET


def governance_dimensions(
    context: CostGovernanceContext,
    *,
    prompt_tokens: int,
    answer_tokens: int,
    total_tokens: int,
    token_source: str,
) -> dict[str, Any]:
    """Build structured governance dimensions around measured/estimated token usage."""

    cache = context.retrieval_cache or {}
    workflow = context.workflow.model_dump() if hasattr(context.workflow, "model_dump") else (
        context.workflow if isinstance(context.workflow, dict) else {}
    )
    budget = _token_budget()
    warnings: list[str] = []
    if total_tokens > budget:
        warnings.append("request_token_budget_exceeded")
    if len(context.tool_calls) > 3:
        warnings.append("tool_call_count_high")
    fragments = selected_prompt_fragments(context)
    return {
        "schema_version": "cost_summary_v1",
        "cost_profile": "request_observation",
        "path_type": context.path_type,
        "model_calls": {
            "route_planner": int(context.route_planner_model_used),
            "final_answer": int(context.final_answer_model_used),
            "extra_reasoning": 0,
        },
        "accounting_scope": {
            "model_calls_are_logical_stage_indicators": True,
            "token_source": token_source,
            "included": ["provided_answer_messages", "final_answer"],
            "not_aggregated": ["embedding", "reranker", "tool_business_api"],
        },
        "tool_call_count": len(context.tool_calls),
        "business_tool_call_count": len(context.tool_calls),
        "rag": {
            "needs_rag": bool(context.citations),
            "hit_count": len(context.citations),
            "cache_hit": bool(cache.get("cache_hit", False)),
            "cacheable": bool(cache.get("cacheable", False)),
            "cache_scope": cache.get("scope"),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "answer": answer_tokens,
            "total": total_tokens,
            "request_budget": budget,
            "remaining_budget": max(0, budget - total_tokens),
        },
        "prompt_fragments": {
            "selected": fragments,
            "fragment_count": len(fragments),
            "fragmentized": True,
        },
        "cache": {
            "retrieval_cache_hit": bool(cache.get("cache_hit", False)),
            "cache_key_present": bool(cache.get("cache_key")),
            "does_not_cache_high_risk_workflow": True,
        },
        "observation_compression": observation_compression(
            context.tool_calls, context.path_type
        ),
        "workflow": {
            "used_langgraph": bool(workflow.get("used_langgraph", False)),
            "workflow_type": workflow.get("workflow_type"),
            "status": workflow.get("status"),
            "hitl_required": bool(workflow.get("pending_action") == "require_human_approval"),
        },
        "degradation": {
            "degraded": context.degraded,
            "cost_threshold_exceeded": bool(warnings),
            "warnings": warnings,
        },
        "safety_boundary": {
            "cost_control_does_not_skip_business_facts": True,
            "cost_control_does_not_skip_hitl": True,
            "not_finops_or_billing_system": True,
        },
    }
