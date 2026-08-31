"""OpenAI-compatible client for prompt-constrained customer-service answers."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel

from api.schemas import TokenUsage
from config.settings import api_key_is_missing, load_project_env
from cost.observer import parse_model_usage


class ModelAnswerResult(BaseModel):
    """Record whether the answer came from the model or a safe fallback."""

    answer: str
    used_model: bool = False
    model_name: str | None = None
    fallback_reason: str | None = None
    usage: TokenUsage | None = None


def create_tool_calling_model() -> Any:
    """Create LangChain's OpenAI-compatible chat model only on the tool route."""

    load_project_env()
    api_key = os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        raise RuntimeError("缺少有效的 AGENT_OPENAI_API_KEY，无法执行模型 Tool Calling。")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B"),
        api_key=api_key,
        base_url=os.getenv(
            "AGENT_OPENAI_BASE_URL",
            "https://api.siliconflow.cn/v1",
        ).rstrip("/"),
        temperature=0.1,
    )


def call_chat_model(
    messages: list[dict[str, str]],
    *,
    fallback_answer: str,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> ModelAnswerResult:
    """Generate an answer from assembled messages and fail closed."""

    load_project_env()
    if os.getenv("AGENT_DISABLE_LLM") == "1" and http_client is None:
        return ModelAnswerResult(answer=fallback_answer, fallback_reason="model_disabled")

    resolved_api_key = api_key if api_key is not None else os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_api_key):
        return ModelAnswerResult(answer=fallback_answer, fallback_reason="model_config_missing")

    resolved_base_url = (base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    resolved_model = model or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    request_kwargs = {
        "headers": {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"},
        "json": {"model": resolved_model, "messages": messages, "temperature": 0.2},
    }
    try:
        if http_client is not None:
            response = http_client.post(f"{resolved_base_url}/chat/completions", **request_kwargs)
        else:
            response = httpx.post(f"{resolved_base_url}/chat/completions", **request_kwargs, timeout=30)
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
        if not answer:
            return ModelAnswerResult(
                answer=fallback_answer,
                model_name=resolved_model,
                fallback_reason="empty_model_answer",
            )
        return ModelAnswerResult(
            answer=answer,
            used_model=True,
            model_name=resolved_model,
            usage=parse_model_usage(payload),
        )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        return ModelAnswerResult(
            answer=fallback_answer,
            model_name=resolved_model,
            fallback_reason=exc.__class__.__name__,
        )
