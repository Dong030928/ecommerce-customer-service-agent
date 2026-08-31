"""Model-assisted clarification wording with backend-authoritative arguments."""

from __future__ import annotations

import json
import os

import httpx

from api.schemas import ChatRequest, ClarificationPlan
from config.settings import api_key_is_missing, load_project_env
from models.classifier_client import parse_classifier_json
from tools.planning import apply_model_clarification_draft


def build_clarification_messages(
    request: ChatRequest,
    plan: ClarificationPlan,
) -> list[dict[str, str]]:
    """Expose only the user message and public tool contract decision to the model."""

    plan_hint = {
        "intent": plan.intent,
        "tool_name": plan.tool_name,
        "missing_required": plan.missing_required,
    }
    return [
        {
            "role": "system",
            "content": (
                "你是电商客服 Agent 的工具澄清规划器，只输出 JSON。"
                "后端已经确定候选工具和缺失字段；你只能生成简短澄清问题，"
                "不能添加 user_id、订单号、退款号等参数，不能替用户选择候选。"
                "输出字段：tool_name、clarification_question、confidence、reason。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"后端计划：{json.dumps(plan_hint, ensure_ascii=False)}\n"
                f"用户消息：{request.user_message}"
            ),
        },
    ]


def plan_clarification_with_model(
    request: ChatRequest,
    authoritative_plan: ClarificationPlan,
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> ClarificationPlan:
    """Let a model draft wording, then retain backend arguments and required fields."""

    load_project_env()
    if os.getenv("AGENT_DISABLE_LLM") == "1" and http_client is None:
        return authoritative_plan.model_copy(update={"source": "backend_fallback"})
    resolved_key = api_key if api_key is not None else os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_key):
        return authoritative_plan.model_copy(update={"source": "backend_fallback"})
    resolved_url = (
        base_url
        or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    ).rstrip("/")
    resolved_model = (
        model
        or os.getenv("AGENT_CLASSIFIER_MODEL")
        or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    )
    request_kwargs = {
        "headers": {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        },
        "json": {
            "model": resolved_model,
            "messages": build_clarification_messages(request, authoritative_plan),
            "temperature": 0,
            "max_tokens": 240,
        },
    }
    try:
        if http_client is not None:
            response = http_client.post(
                f"{resolved_url}/chat/completions",
                **request_kwargs,
            )
        else:
            response = httpx.post(
                f"{resolved_url}/chat/completions",
                **request_kwargs,
                timeout=30,
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = parse_classifier_json(str(content))
        if payload is None:
            return authoritative_plan.model_copy(update={"source": "backend_fallback"})
        return apply_model_clarification_draft(
            authoritative_plan,
            payload,
            model_name=resolved_model,
        )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return authoritative_plan.model_copy(update={"source": "backend_fallback"})
