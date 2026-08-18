"""模型客户端层，封装 OpenAI-compatible 聊天模型调用和错误边界。"""

from __future__ import annotations

import os

import httpx

from config.settings import api_key_is_missing, load_project_env


def call_chat_model(
    user_message: str,
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""

    load_project_env()
    resolved_api_key = api_key if api_key is not None else os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(resolved_api_key):
        raise RuntimeError(
            "缺少有效的 AGENT_OPENAI_API_KEY。/chat 入口已收到请求，但不能把模型配置失败伪装成客服回答。"
        )

    resolved_base_url = (base_url or os.getenv("AGENT_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
    resolved_model = model or os.getenv("AGENT_OPENAI_MODEL", "Qwen/Qwen3-8B")
    request_kwargs = {
        "headers": {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"},
        "json": {
            "model": resolved_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是电商平台的客服 Agent。当前版本只负责普通聊天，"
                        "不能承诺优惠、退款、物流或售后处理结果。"
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        },
    }
    if http_client is not None:
        response = http_client.post(f"{resolved_base_url}/chat/completions", **request_kwargs)
    else:
        response = httpx.post(f"{resolved_base_url}/chat/completions", **request_kwargs, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]
