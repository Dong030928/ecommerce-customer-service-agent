"""Minimal agent orchestration for the first project version."""

from __future__ import annotations

import httpx

from api.schemas import ChatRequest, ChatResponse
from models.llm_client import call_chat_model


class CustomerServiceAgent:
    """Minimal customer service agent with a stable chat contract."""

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

        # Business tools and policy retrieval are intentionally deferred to later versions.
        answer = call_chat_model(
            request.user_message,
            http_client=self._chat_http_client,
            api_key=self._chat_api_key,
            base_url=self._chat_base_url,
            model=self._chat_model_name,
        )
        session_state = {
            "agent_version": "0.1.0",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            "next_gap": "当前版本尚未接入业务工具、知识检索、多轮记忆和高风险工作流。",
        }

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            session_state=session_state,
        )
