"""Load, select, and render prompt fragments from a registry."""

from __future__ import annotations

import json

from api.schemas import ChatRequest, Intent, IntentResult, PromptFragment
from config.settings import PROMPT_REGISTRY_PATH


def load_prompt_registry() -> list[PromptFragment]:
    """Load and validate project prompt fragments from JSON."""

    with PROMPT_REGISTRY_PATH.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("prompt_registry.json 顶层必须是数组。")
    return [PromptFragment.model_validate(item) for item in payload]


def select_prompt_fragments(
    intent: Intent,
    registry: list[PromptFragment],
) -> list[PromptFragment]:
    """Select enabled fragments for an intent and order by priority."""

    selected = [
        fragment
        for fragment in registry
        if fragment.enabled
        and ("all" in fragment.applies_to or intent in fragment.applies_to)
    ]
    return sorted(selected, key=lambda fragment: fragment.priority, reverse=True)


def render_prompt_template(
    request: ChatRequest,
    intent_result: IntentResult,
    fragments: list[PromptFragment],
) -> list[dict[str, str]]:
    """Render selected fragments without disclosing trusted identity values."""

    fragment_text = "\n\n".join(
        f"[{fragment.fragment_id} | priority={fragment.priority}]\n{fragment.content}"
        for fragment in fragments
    )
    system_message = (
        "你是电商平台的客服 Agent。当前版本使用 Prompt Registry 管理规则片段，"
        "并观察每轮 Prompt 与回答的 token 消耗。\n"
        "请严格按照片段优先级回答；高优先级片段覆盖低优先级片段。\n\n"
        f"{fragment_text}"
    )
    user_message = (
        "平台运行时事实说明：\n"
        "- 用户身份、昵称、会员等级和风险等级由平台在本地可信上下文中维护，"
        "本轮不向外部模型披露具体值。\n"
        "- 如果回答依赖未提供给模型的运行时事实，应明确说明需要由平台系统核实。\n"
        "\n"
        f"粗意图：{intent_result.intent}\n"
        f"粗意图说明：{intent_result.explanation}\n"
        "\n"
        "用户原话：\n"
        f"{request.user_message}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
