"""Structured intent classifier with a model fallback."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from api.schemas import Intent, IntentResult
from config.settings import api_key_is_missing, load_project_env


def build_classifier_messages(user_message: str) -> list[dict[str, str]]:
    """Build messages that constrain the classifier to a stable JSON contract."""

    allowed_intents = ", ".join(Intent.__args__)  # type: ignore[attr-defined]
    return [
        {
            "role": "system",
            "content": (
                "你是电商客服 Agent 的轻量意图分类器。只输出 JSON，不要输出 Markdown。"
                "你只能判断用户消息的大类，不能执行工具、批准退款或承诺售后动作。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请把下面用户消息分成一个粗意图。\n"
                f"允许的 intent 只能是：{allowed_intents}\n"
                "输出 JSON 格式：{\"intent\": string, \"confidence\": number, "
                "\"explanation\": string}\n\n"
                f"用户消息：{user_message}"
            ),
        },
    ]


def parse_classifier_json(content: str) -> dict[str, Any] | None:
    """Parse plain JSON and JSON wrapped in a Markdown code fence."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def classify_intent_with_model(
    user_message: str,
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> IntentResult | None:
    """Use a lightweight model when deterministic rules are inconclusive."""

    load_project_env()
    if os.getenv("AGENT_DISABLE_LLM") == "1" and http_client is None:
        return None

    resolved_api_key = api_key if api_key is not None else os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_api_key):
        return None

    resolved_base_url = (base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    resolved_model = model or os.getenv("AGENT_CLASSIFIER_MODEL") or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    request_kwargs = {
        "headers": {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"},
        "json": {"model": resolved_model, "messages": build_classifier_messages(user_message)},
    }

    try:
        if http_client is not None:
            response = http_client.post(f"{resolved_base_url}/chat/completions", **request_kwargs)
        else:
            response = httpx.post(f"{resolved_base_url}/chat/completions", **request_kwargs, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = parse_classifier_json(content)
        if payload is None:
            return None
        return IntentResult(
            intent=payload.get("intent", "unknown"),
            source="classifier",
            confidence=float(payload.get("confidence", 0.7)),
            matched_keywords=[],
            explanation=str(payload.get("explanation") or "分类模型给出粗意图兜底。"),
        )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValidationError, ValueError):
        return None
