"""Build a dual-channel Runtime Context that user text cannot overwrite."""

from __future__ import annotations

import re

from api.schemas import ChatRequest, RuntimeContextView, ToolCallRecord
from tools.runtime_context import public_runtime_context


VIP_CLAIM_TERMS = ("我是VIP", "我是 VIP", "我是黑卡", "我是尊贵会员")
VIP_CLAIM_PATTERN = re.compile(r"(?:我是|也是)\s*VIP", re.IGNORECASE)
USER_ID_CLAIM_PATTERN = re.compile(
    r"(?:我是|账号是|用户是)\s*([Uu][A-Za-z0-9_-]{2,})"
)


def user_claims_vip(message: str) -> bool:
    """Detect a membership claim without treating it as trusted identity data."""

    return any(term in message for term in VIP_CLAIM_TERMS) or bool(
        VIP_CLAIM_PATTERN.search(message)
    )


def user_claims_another_identity(message: str, runtime_user_id: str) -> bool:
    """Detect a textual user-id claim that conflicts with the authenticated user."""

    match = USER_ID_CLAIM_PATTERN.search(message)
    if match is None:
        return False
    return match.group(1).casefold() != runtime_user_id.strip().casefold()


def build_runtime_context_view(request: ChatRequest) -> RuntimeContextView:
    """Separate low-risk model context from backend-only authorization signals."""

    authenticated = bool(request.runtime_user_id.strip())
    member_level = (request.runtime_member_level or "unknown").strip().lower()
    risk_level = (request.runtime_risk_level or "unknown").strip().lower()
    permissions = ["read_own_order", "ask_after_sale"] if authenticated else []
    if member_level == "vip":
        permissions.append("vip_service")

    conflict_notes: list[str] = []
    if user_claims_vip(request.user_message) and member_level != "vip":
        conflict_notes.append(
            "用户文本自称 VIP，但可信登录态未确认；会员权益仍以 Runtime Context 为准。"
        )
    if user_claims_another_identity(request.user_message, request.runtime_user_id):
        conflict_notes.append(
            "用户文本中的身份声明与可信登录态冲突，声明不会覆盖系统用户身份。"
        )

    return RuntimeContextView(
        trusted_for_model={
            "authenticated": authenticated,
            "nickname": request.runtime_nickname,
            "member_level": member_level,
            "page_context": public_runtime_context(request),
        },
        system_only={
            "identity_source": "trusted_runtime",
            "user_id_present": authenticated,
            "risk_level": risk_level,
            "permissions": permissions,
            "raw_user_id_exposed": False,
        },
        conflict_notes=conflict_notes,
        permission_decision={
            "allowed": True,
            "reason": "本轮尚未触发需要订单归属校验的工具。",
            "checked": False,
        },
    )


def apply_permission_decision(
    view: RuntimeContextView,
    tool_calls: list[ToolCallRecord],
) -> RuntimeContextView:
    """Derive a public authorization result from controlled order observations."""

    order_calls = [
        record
        for record in tool_calls
        if record.action.tool_name in {"get_order_status", "get_order_logistics"}
    ]
    if not order_calls:
        return view
    failed = next(
        (record for record in order_calls if record.observation.status != "success"),
        None,
    )
    if failed is not None:
        return view.model_copy(
            update={
                "permission_decision": {
                    "allowed": False,
                    "reason": failed.observation.error_code
                    or "trusted_order_check_failed",
                    "checked": True,
                }
            }
        )
    return view.model_copy(
        update={
            "permission_decision": {
                "allowed": True,
                "reason": "订单读取已绑定可信 Runtime 用户身份并由业务系统放行。",
                "checked": True,
            }
        }
    )


def build_member_context_answer(
    request: ChatRequest,
    view: RuntimeContextView,
) -> str:
    """Answer membership questions from trusted state, never from user claims."""

    member_level = str(view.trusted_for_model.get("member_level") or "unknown")
    if user_claims_vip(request.user_message) and member_level != "vip":
        if member_level == "unknown":
            return "可信 Runtime Context 尚未提供会员等级，我不能按聊天中的自称把你当作 VIP。"
        return f"可信登录态显示当前会员等级为 {member_level}，不能按聊天中的自称提升为 VIP。"
    if member_level == "vip":
        return "可信登录态确认当前会员等级为 VIP，可以按 VIP 服务口径继续咨询。"
    if member_level == "unknown":
        return "可信 Runtime Context 尚未提供会员等级，我会按普通权益咨询处理。"
    return f"可信登录态显示当前会员等级为 {member_level}，我会按该等级回答。"
