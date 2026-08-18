"""Agent orchestration and the first explicit customer-service boundary."""

from __future__ import annotations

import httpx

from api.schemas import ChatRequest, ChatResponse
from models.llm_client import call_chat_model


def build_customer_service_messages(request: ChatRequest) -> list[dict[str, str]]:
    """Build model messages that declare the service identity and factual limits."""

    system_message = (
        "你是电商平台的第一版 AI 客服，请用自然、耐心的客服语气回答用户问题。"
        "当前版本只接入了大模型聊天能力，还没有接入平台活动规则、订单物流、"
        "退款条件、售后流程或业务工具。缺少业务事实时必须说明边界，不能编造或承诺处理结果。"
    )
    user_message = "用户问题：\n" + request.user_message
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


class CustomerServiceAgent:
    """First LLM customer-service agent with an explicit business boundary."""

    def __init__(
        self,
        *,
        chat_http_client: httpx.Client | None = None,
        chat_api_key: str | None = None,
        chat_base_url: str | None = None,
        chat_model_name: str | None = None,
    ) -> None:
        """Initialize the agent and allow a test HTTP client to be injected."""

        self._message_count_by_session: dict[str, int] = {}
        self._chat_http_client = chat_http_client
        self._chat_api_key = chat_api_key
        self._chat_base_url = chat_base_url
        self._chat_model_name = chat_model_name

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request and expose the current minimal session state."""

        self._message_count_by_session[request.session_id] = (
            self._message_count_by_session.get(request.session_id, 0) + 1
        )
        message_count = self._message_count_by_session[request.session_id]

        messages = build_customer_service_messages(request)
        answer = call_chat_model(
            messages,
            http_client=self._chat_http_client,
            api_key=self._chat_api_key,
            base_url=self._chat_base_url,
            model=self._chat_model_name,
        )
        reasoning_summary = [
            "后端接收 ChatRequest，并把用户问题包装成受控的客服 messages。",
            "模型负责生成自然语言回答，但当前没有活动、订单、物流或退款事实来源。",
            "当前链路没有业务 Action 和业务系统 Observation。",
            "模型回答不能被业务系统当作实际处理结果。",
        ]
        session_state = {
            "agent_version": "0.2.0",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            "llm_customer_boundary": {
                "answer_source": "llm_only",
                "promotion_rules": "not_connected",
                "order_logistics": "not_connected",
                "refund_policy": "not_connected",
                "business_tools": "not_connected",
            },
            "next_gap": "AI 能生成客服话术，但还不能识别和结构化表达用户的业务意图。",
        }

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            reasoning_summary=reasoning_summary,
            session_state=session_state,
        )
