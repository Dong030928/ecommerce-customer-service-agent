"""OpenAI-compatible client for low-confidence RoutePlan drafts."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from api.schemas import Intent, ToolCandidate
from config.settings import api_key_is_missing, load_project_env
from models.classifier_client import parse_classifier_json


def build_task_planner_messages(
    user_message: str,
    candidates: list[ToolCandidate],
) -> list[dict[str, str]]:
    """Expose only the user question and public tool summaries to the model."""

    allowed_intents = ", ".join(Intent.__args__)  # type: ignore[attr-defined]
    candidate_payload = [candidate.model_dump() for candidate in candidates]
    return [
        {
            "role": "system",
            "content": (
                "你是电商客服 Agent 的入口任务规划器。只输出 JSON，不输出 Markdown。"
                "你只生成短小的路由草案，不能执行工具、读取用户身份、批准退款或"
                "生成隐藏推理过程。required_tools 只能来自候选工具列表。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"允许的 intent：{allowed_intents}\n"
                "输出字段：intent、needs_rag、needs_business_tools、rag_query、"
                "confidence、intents、entity_refs、required_context、required_tools、"
                "knowledge_domains、has_realtime_fact、risk_level、requires_workflow、"
                "fallback_policy。\n"
                "稳定政策和商品知识需要 RAG；订单、物流、库存、价格和退款进度等"
                "变化事实需要只读工具；写操作必须 requires_workflow=true。\n"
                "候选工具："
                f"{json.dumps(candidate_payload, ensure_ascii=False)}\n"
                f"用户问题：{user_message}"
            ),
        },
    ]


class TaskPlannerModelClient:
    """Request a structured draft only when deterministic rules are uncertain."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    def plan(
        self,
        user_message: str,
        candidates: list[ToolCandidate],
    ) -> dict[str, Any] | None:
        load_project_env()
        if os.getenv("AGENT_DISABLE_LLM") == "1" and self._http_client is None:
            return None
        api_key = (
            self._api_key
            if self._api_key is not None
            else os.getenv("AGENT_OPENAI_API_KEY")
        )
        if api_key_is_missing(api_key):
            return None
        base_url = (
            self._base_url
            or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        ).rstrip("/")
        model = (
            self._model
            or os.getenv("AGENT_CLASSIFIER_MODEL")
            or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
        )
        request_kwargs = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": model,
                "messages": build_task_planner_messages(user_message, candidates),
                "max_tokens": 240,
                "temperature": 0,
            },
        }
        try:
            if self._http_client is not None:
                response = self._http_client.post(
                    f"{base_url}/chat/completions",
                    **request_kwargs,
                )
            else:
                response = httpx.post(
                    f"{base_url}/chat/completions",
                    **request_kwargs,
                    timeout=15,
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_classifier_json(str(content))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None
