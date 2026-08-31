"""LangChain model Tool Calling with backend-controlled Action execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from api.schemas import (
    ChatRequest,
    Intent,
    ToolAction,
    ToolCallRecord,
    ToolObservation,
)
from config.settings import TOOL_CALLING_RECURSION_LIMIT
from models.llm_client import create_tool_calling_model
from tools.contracts import TOOL_SPECS
from tools.langchain_tools import build_langchain_tools, tool_action_reason
from tools.planning import missing_reference_prompt
from tools.tool_runtime import ToolRuntime


@dataclass(frozen=True)
class ToolCallingOutcome:
    answer: str
    tool_calls: list[ToolCallRecord]
    state: dict[str, Any]
    used_model: bool
    model_name: str | None = None
    error: str | None = None


def _model_label(model: Any) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    return model.__class__.__name__


def _tool_records(messages: list[Any]) -> list[ToolCallRecord]:
    observations_by_id: dict[str, ToolObservation] = {}
    for message in messages:
        if message.__class__.__name__ != "ToolMessage":
            continue
        try:
            observations_by_id[str(message.tool_call_id)] = (
                ToolObservation.model_validate_json(str(message.content))
            )
        except (AttributeError, TypeError, ValueError):
            continue

    records: list[ToolCallRecord] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            observation = observations_by_id.get(str(tool_call.get("id") or ""))
            if observation is None:
                continue
            tool_name = str(tool_call.get("name") or "")
            records.append(
                ToolCallRecord(
                    action=ToolAction(
                        tool_name=tool_name,
                        arguments=dict(tool_call.get("args") or {}),
                        reason=tool_action_reason(tool_name),
                    ),
                    observation=observation,
                )
            )
    return records


def _answer_from_observations(records: list[ToolCallRecord]) -> str:
    """Ground the user-facing answer in sanitized observations, not free wording."""

    successful = [
        record.observation.summary
        for record in records
        if record.observation.status == "success"
    ]
    if successful:
        return "我通过实时业务工具查到：" + " ".join(successful)
    failed = [record.observation.summary for record in records]
    if failed:
        return "这次工具没有查到可用事实：" + " ".join(failed)
    return "模型没有生成可执行的只读工具调用，因此我不能猜测实时业务状态。"


class ToolCallingService:
    """Run a bounded tool loop while keeping identity inside the backend runtime."""

    def __init__(
        self,
        *,
        runtime: ToolRuntime | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._runtime = runtime or ToolRuntime()
        self._model_factory = model_factory or create_tool_calling_model

    def run(self, request: ChatRequest, intent: Intent) -> ToolCallingOutcome:
        missing_prompt = missing_reference_prompt(intent, request.user_message)
        available_tools = [spec.model_dump() for spec in TOOL_SPECS.values()]
        if missing_prompt:
            return ToolCallingOutcome(
                answer=missing_prompt,
                tool_calls=[],
                state={
                    "create_agent": False,
                    "skip_reason": "missing_required_business_reference",
                    "available_tools": available_tools,
                    "message_types": [],
                    "answer_source": "deterministic_parameter_guard",
                },
                used_model=False,
            )
        try:
            from langchain.agents import create_agent

            model = self._model_factory()
            tools = build_langchain_tools(request, self._runtime)
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=(
                    "你是电商客服 Agent。实时订单、物流、商品库存、价格或退款进度"
                    "必须通过给定只读工具核验。只能提交工具契约声明的参数，不能提交、"
                    "猜测或覆盖 user_id；当前用户身份由后端从可信 Runtime Context 注入。"
                    "不要调用写操作，也不要根据知识库猜实时状态。"
                ),
            )
            result = agent.invoke(
                {"messages": [{"role": "user", "content": request.user_message}]},
                config={"recursion_limit": TOOL_CALLING_RECURSION_LIMIT},
            )
            messages = list(result.get("messages") or [])
            records = _tool_records(messages)
            model_name = _model_label(model)
            return ToolCallingOutcome(
                answer=_answer_from_observations(records),
                tool_calls=records,
                state={
                    "create_agent": True,
                    "available_tools": available_tools,
                    "message_types": [
                        message.__class__.__name__ for message in messages
                    ],
                    "answer_source": "sanitized_tool_observation",
                    "model_final_wording_used": False,
                },
                used_model=True,
                model_name=model_name,
            )
        except (ImportError, RuntimeError) as exc:
            return ToolCallingOutcome(
                answer=(
                    "实时业务工具当前不可用，不能据此猜测订单、物流、库存或退款状态。"
                ),
                tool_calls=[],
                state={
                    "create_agent": False,
                    "skip_reason": "tool_calling_unavailable",
                    "available_tools": available_tools,
                    "message_types": [],
                    "answer_source": "safe_tool_fallback",
                },
                used_model=False,
                error=exc.__class__.__name__,
            )
        except Exception as exc:  # Provider/LangChain errors share no stable base class.
            return ToolCallingOutcome(
                answer="实时业务工具调用失败，我不能编造本轮业务事实，请稍后重试。",
                tool_calls=[],
                state={
                    "create_agent": True,
                    "skip_reason": "tool_calling_failed",
                    "available_tools": available_tools,
                    "message_types": [],
                    "answer_source": "safe_tool_fallback",
                },
                used_model=True,
                error=exc.__class__.__name__,
            )
