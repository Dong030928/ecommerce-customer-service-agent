"""Token usage parsing, local estimation, and cost observation."""

from __future__ import annotations

import os
from typing import Any, Literal

from api.schemas import CostSummary, TokenUsage
from config.settings import DEFAULT_INPUT_CNY_PER_1K, DEFAULT_OUTPUT_CNY_PER_1K


def estimate_tokens(text: str) -> int:
    """Estimate token trends when a model provider omits usage data."""

    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars // 2)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> tuple[int, int]:
    """Estimate message tokens and return the observed character count."""

    text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
    return estimate_tokens(text), len(text)


def _read_usage_int(usage: dict[str, Any], *names: str) -> int | None:
    """Read an integer from common OpenAI-compatible usage field names."""

    for name in names:
        raw_value = usage.get(name)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def parse_model_usage(payload: dict[str, Any]) -> TokenUsage | None:
    """Normalize model-provider usage without trusting incomplete counters."""

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = _read_usage_int(usage, "prompt_tokens", "input_tokens")
    answer_tokens = _read_usage_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "answer_tokens",
    )
    total_tokens = _read_usage_int(usage, "total_tokens")
    if answer_tokens is None and prompt_tokens is not None and total_tokens is not None:
        answer_tokens = max(0, total_tokens - prompt_tokens)
    if prompt_tokens is None and answer_tokens is not None and total_tokens is not None:
        prompt_tokens = max(0, total_tokens - answer_tokens)
    if prompt_tokens is None or answer_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = prompt_tokens + answer_tokens

    completion_details = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    details = {
        "reasoning_tokens": (
            _read_usage_int(completion_details, "reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        ),
        "cached_tokens": (
            _read_usage_int(prompt_details, "cached_tokens")
            if isinstance(prompt_details, dict)
            else None
        ),
        "prompt_cache_hit_tokens": _read_usage_int(usage, "prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": _read_usage_int(usage, "prompt_cache_miss_tokens"),
    }
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        total_tokens=total_tokens,
        details={key: value for key, value in details.items() if value is not None},
    )


def read_price_per_1k(name: str, default: float) -> float:
    """Read a non-negative per-1K-token estimate or use the project default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value >= 0 else default


def build_cost_summary(
    messages: list[dict[str, str]],
    answer: str,
    usage: TokenUsage | None = None,
) -> CostSummary:
    """Build a public cost summary from provider usage or a local estimate."""

    estimated_prompt_tokens, context_chars = estimate_messages_tokens(messages)
    if usage is not None:
        prompt_tokens = usage.prompt_tokens
        answer_tokens = usage.answer_tokens
        total_tokens = usage.total_tokens
        usage_details = usage.details
        token_source: Literal["model_usage", "local_estimate"] = "model_usage"
        pricing_note = "优先使用模型平台 usage；估算金额不替代平台真实账单。"
    else:
        prompt_tokens = estimated_prompt_tokens
        answer_tokens = estimate_tokens(answer)
        total_tokens = prompt_tokens + answer_tokens
        usage_details = {}
        token_source = "local_estimate"
        pricing_note = "模型平台未返回 usage，使用本地 token 估算观察趋势；真实账单以平台为准。"

    input_price = read_price_per_1k("AGENT_INPUT_CNY_PER_1K", DEFAULT_INPUT_CNY_PER_1K)
    output_price = read_price_per_1k("AGENT_OUTPUT_CNY_PER_1K", DEFAULT_OUTPUT_CNY_PER_1K)
    input_cost = prompt_tokens / 1000 * input_price
    output_cost = answer_tokens / 1000 * output_price
    return CostSummary(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        total_tokens=total_tokens,
        token_source=token_source,
        usage_details=usage_details,
        estimated_input_cost_cny=round(input_cost, 6),
        estimated_output_cost_cny=round(output_cost, 6),
        estimated_total_cost_cny=round(input_cost + output_cost, 6),
        context_chars=context_chars,
        pricing_note=pricing_note,
    )
