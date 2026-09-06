"""Short-lived session memory with explicit write and exclusion policies."""

from __future__ import annotations

import re
from threading import RLock

from api.schemas import (
    ChatRequest,
    Intent,
    MemoryDecision,
    SessionMemorySnapshot,
    ToolCallRecord,
)
from tools.planning import extract_order_id


ORDER_REFERENCES = ("刚才那个", "刚刚那个", "上一个订单", "前面那个订单")


def references_recent_order(message: str) -> bool:
    """Return whether a message refers to a previously verified order."""

    return any(term in message for term in ORDER_REFERENCES)


def excluded_categories(message: str) -> list[str]:
    """Classify content that must never become reusable session context."""

    categories: list[str] = []
    if re.search(r"1[3-9]\d{9}", message):
        categories.append("phone_number")
    if any(term in message for term in ("收货地址", "家庭住址", "身份证", "银行卡", "支付密码")):
        categories.append("private_identity_or_address")
    if any(
        term in message
        for term in (
            "resume_token",
            "审批令牌",
            "恢复令牌",
            "主管同意",
            "审批通过",
        )
    ):
        categories.append("approval_or_resume_claim")
    if any(term in message for term in ("系统提示词", "hidden reasoning", "内部策略", "思维链")):
        categories.append("system_or_reasoning_request")
    if any(term in message for term in ("我是会员", "我是VIP", "我是管理员", "我是主管")):
        categories.append("unverified_identity_claim")
    return categories


def low_risk_preference(message: str) -> tuple[str, str] | None:
    """Extract only a small allowlist of low-risk, session-scoped preferences."""

    if not any(term in message for term in ("喜欢", "偏好", "以后", "推荐")):
        return None
    for color in ("黑色", "白色", "蓝色"):
        if color in message:
            return "preferred_color", color
    return None


class SessionMemoryStore:
    """Keep low-risk memory isolated by both session and trusted runtime user."""

    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], SessionMemorySnapshot] = {}
        self._lock = RLock()

    @staticmethod
    def _key(request: ChatRequest) -> tuple[str, str]:
        return request.session_id, request.runtime_user_id

    def snapshot(self, request: ChatRequest) -> SessionMemorySnapshot:
        """Return a copy so callers cannot mutate the store without policy checks."""

        with self._lock:
            snapshot = self._snapshots.setdefault(
                self._key(request), SessionMemorySnapshot()
            )
            return snapshot.model_copy(deep=True)

    def enrich_request(self, request: ChatRequest) -> tuple[ChatRequest, bool]:
        """Resolve a recent-order reference through trusted runtime context only."""

        if extract_order_id(request.user_message) or not references_recent_order(
            request.user_message
        ):
            return request, False
        snapshot = self.snapshot(request)
        if not snapshot.last_order_id:
            return request, False
        context = dict(request.runtime_context or {})
        if context.get("relatedOrderNo"):
            return request, False
        context["relatedOrderNo"] = snapshot.last_order_id
        return request.model_copy(update={"runtime_context": context}), True

    def update(
        self,
        *,
        request: ChatRequest,
        intent: Intent,
        tool_calls: list[ToolCallRecord],
    ) -> tuple[list[MemoryDecision], SessionMemorySnapshot]:
        """Apply the allowlist policy after trusted tools have finished."""

        with self._lock:
            memory = self._snapshots.setdefault(
                self._key(request), SessionMemorySnapshot()
            )
            decisions: list[MemoryDecision] = []
            for category in excluded_categories(request.user_message):
                if category not in memory.excluded_items:
                    memory.excluded_items.append(category)
                decisions.append(
                    MemoryDecision(
                        key=category,
                        accepted=False,
                        reason="该内容属于隐私、高风险审批、内部信息或未验证身份声明，不写入会话记忆。",
                    )
                )

            verified_order_id: str | None = None
            verified_product_name: str | None = None
            for record in tool_calls:
                observation = record.observation
                if observation.status != "success":
                    continue
                facts = observation.facts
                if record.action.tool_name in {
                    "get_order_status",
                    "get_order_logistics",
                }:
                    value = facts.get("order_id")
                    if value:
                        verified_order_id = str(value)
                if record.action.tool_name == "get_product_inventory":
                    value = facts.get("name") or facts.get("sku")
                    if value:
                        verified_product_name = str(value)

            explicit_order_id = extract_order_id(request.user_message)
            if verified_order_id:
                memory.last_order_id = verified_order_id
                decisions.append(
                    MemoryDecision(
                        key="last_order_id",
                        value=verified_order_id,
                        accepted=True,
                        reason="订单已由受控业务工具按可信用户身份校验，可用于当前会话后续消歧。",
                        ttl="session",
                    )
                )
            elif explicit_order_id:
                decisions.append(
                    MemoryDecision(
                        key="last_order_id",
                        accepted=False,
                        reason="订单未通过受控业务工具校验，不写入最近订单记忆。",
                    )
                )

            if verified_product_name:
                memory.last_product_name = verified_product_name
                decisions.append(
                    MemoryDecision(
                        key="last_product_name",
                        value=verified_product_name,
                        accepted=True,
                        reason="商品由只读业务工具确认，可在当前会话辅助后续商品咨询。",
                        ttl="session",
                    )
                )

            preference = low_risk_preference(request.user_message)
            if preference:
                key, value = preference
                memory.low_risk_preferences[key] = value
                decisions.append(
                    MemoryDecision(
                        key=f"low_risk_preferences.{key}",
                        value=value,
                        accepted=True,
                        reason="颜色属于低风险偏好，仅在当前会话短期保留。",
                        ttl="session",
                    )
                )

            memory.recent_intent = intent
            decisions.append(
                MemoryDecision(
                    key="recent_intent",
                    value=intent,
                    accepted=True,
                    reason="最近意图仅用于当前会话的多轮消歧。",
                    ttl="session",
                )
            )
            return decisions, memory.model_copy(deep=True)
