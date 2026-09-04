"""High-risk after-sale eligibility checks built from read-only evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from api.schemas import (
    ChatRequest,
    Citation,
    HighRiskActionType,
    HighRiskAssessment,
    ToolAction,
    ToolCallRecord,
)
from hooks.manager import HookManager
from tools.tool_runtime import ToolRuntime


BLOCKED_WRITE_ACTIONS = [
    "create_refund",
    "approve_refund",
    "cancel_order",
    "create_compensation",
]


@dataclass(frozen=True)
class AfterSaleBoundaryResult:
    assessment: HighRiskAssessment
    tool_calls: list[ToolCallRecord]


def detect_high_risk_action(message: str) -> HighRiskActionType:
    """Classify only the action subtype; routing remains TaskPlanner's job."""

    if any(term in message for term in ["赔付", "赔偿", "补偿"]):
        return "compensation"
    if "取消订单" in message or "取消这个订单" in message:
        return "cancel"
    if "退货" in message:
        return "return"
    if any(term in message for term in ["退款", "退钱"]):
        return "refund"
    return "unknown"


def clarification_assessment(action_type: HighRiskActionType) -> HighRiskAssessment:
    """Require an explicit order before any per-user business read."""

    return HighRiskAssessment(
        action_type=action_type,
        eligibility_status="needs_clarification",
        evidence_checklist=[],
        reasons=["高风险售后必须先确认订单号，不能只凭自然语言执行写操作。"],
        blocked_write_actions=BLOCKED_WRITE_ACTIONS,
    )


def _normalized(value: object) -> str:
    return str(value or "").strip().upper()


def _signed_days(value: object, *, today: date | None = None) -> int | None:
    """Return whole calendar days since delivery, rejecting malformed values."""

    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        delivered = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            delivered = date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return ((today or date.today()) - delivered).days


def _has_return_reason(message: str) -> bool:
    return any(
        term in message
        for term in [
            "七天无理由",
            "7天无理由",
            "不想要",
            "不合适",
            "质量问题",
            "破损",
            "损坏",
            "错发",
            "少发",
        ]
    )


class AfterSalePolicyService:
    """Read trusted facts and return an assessment, never a mutation command."""

    def __init__(self, runtime: ToolRuntime | None = None) -> None:
        self._runtime = runtime or ToolRuntime()

    def _read(
        self,
        tool_name: str,
        order_id: str,
        request: ChatRequest,
        hooks: HookManager,
    ) -> ToolCallRecord:
        action = ToolAction(
            tool_name=tool_name,
            arguments={"order_id": order_id},
            reason=(
                "高风险资格判断必须读取当前用户的订单事实。"
                if tool_name == "get_order_status"
                else "高风险资格判断必须核验订单的物流状态。"
            ),
        )
        observation = self._runtime.execute(action, request, hooks)
        return ToolCallRecord(
            action=action,
            observation=observation,
            attempts=observation.attempts,
        )

    def read_order(
        self,
        order_id: str,
        request: ChatRequest,
        hooks: HookManager,
    ) -> ToolCallRecord:
        """Expose the fixed workflow order-read node without exposing runtime internals."""

        return self._read("get_order_status", order_id, request, hooks)

    def read_logistics(
        self,
        order_id: str,
        request: ChatRequest,
        hooks: HookManager,
    ) -> ToolCallRecord:
        """Expose the fixed workflow logistics-read node."""

        return self._read("get_order_logistics", order_id, request, hooks)

    def assess(
        self,
        *,
        request: ChatRequest,
        order_id: str,
        action_type: HighRiskActionType,
        policy_basis: list[Citation],
        hooks: HookManager,
    ) -> AfterSaleBoundaryResult:
        """Assess application eligibility from safe observations and RAG evidence."""

        order_call = self.read_order(order_id, request, hooks)
        logistics_call = self.read_logistics(order_id, request, hooks)
        tool_calls = [order_call, logistics_call]
        assessment = self.assess_from_evidence(
            order_id=order_id,
            action_type=action_type,
            policy_basis=policy_basis,
            order_call=order_call,
            logistics_call=logistics_call,
            user_message=request.user_message,
        )
        return AfterSaleBoundaryResult(
            assessment=assessment,
            tool_calls=tool_calls,
        )

    def assess_from_evidence(
        self,
        *,
        order_id: str,
        action_type: HighRiskActionType,
        policy_basis: list[Citation],
        order_call: ToolCallRecord | None,
        logistics_call: ToolCallRecord | None,
        user_message: str = "",
    ) -> HighRiskAssessment:
        """Evaluate evidence collected by explicit workflow nodes."""

        tool_calls = [
            record for record in [order_call, logistics_call] if record is not None
        ]
        failed = [
            record.observation
            for record in tool_calls
            if record.observation.status != "success"
        ]
        evidence = ["订单状态", "支付状态", "物流状态"]
        if policy_basis:
            evidence.append("售后政策引用")
        if failed or order_call is None or logistics_call is None:
            return HighRiskAssessment(
                action_type=action_type,
                order_id=order_id,
                eligibility_status="blocked",
                evidence_checklist=evidence,
                policy_basis=policy_basis,
                reasons=["未取得完整可信的订单或物流事实，不能判断售后资格。"],
                blocked_write_actions=BLOCKED_WRITE_ACTIONS,
            )

        order_facts = order_call.observation.facts
        logistics_facts = logistics_call.observation.facts
        order_status = _normalized(order_facts.get("order_status"))
        fulfillment_status = _normalized(order_facts.get("fulfillment_status"))
        payment_status = _normalized(order_facts.get("payment_status"))
        logistics_status = _normalized(logistics_facts.get("logistics_status"))
        not_shipped = logistics_status in {
            "",
            "NOT_SHIPPED",
            "PENDING_SHIPMENT",
            "PENDING",
            "待发货",
        } and order_status not in {"SHIPPED", "DELIVERED", "COMPLETED", "已发货", "已完成"}
        not_shipped = not_shipped and fulfillment_status not in {
            "SHIPPED",
            "IN_TRANSIT",
            "DELIVERED",
            "SIGNED",
        }
        signed = logistics_status in {"SIGNED", "DELIVERED", "已签收"} or order_status in {
            "DELIVERED",
            "COMPLETED",
            "已完成",
        }

        if not policy_basis:
            status = "blocked"
            reasons = ["未检索到可靠的售后政策依据，不能仅凭订单状态承诺资格。"]
        elif action_type == "refund":
            eligible = payment_status == "PAID" and not_shipped
            status = "eligible_for_application" if eligible else "not_eligible"
            reasons = [
                (
                    "订单已支付且未发货，只能判断为可发起退款申请，仍不得自动退款。"
                    if eligible
                    else "订单支付、履约或物流状态不满足未发货退款的基础条件。"
                )
            ]
        elif action_type == "return":
            evidence.extend(["签收时间", "商品可退属性", "退货原因"])
            delivered_at = (
                logistics_facts.get("delivered_at")
                or order_facts.get("delivered_at")
            )
            days_since_signed = _signed_days(delivered_at)
            returnable = order_facts.get("returnable")
            has_reason = _has_return_reason(user_message)
            within_window = (
                days_since_signed is not None and 0 <= days_since_signed <= 7
            )
            eligible = signed and within_window and returnable is True and has_reason
            if eligible:
                status = "eligible_for_application"
                reasons = [
                    f"订单已签收 {days_since_signed} 天，商品支持退货且原因符合基础条件；只能准备申请，不得自动批准。"
                ]
            elif not signed:
                status = "not_eligible"
                reasons = ["订单尚未形成可信签收事实，不能进入签收后退货流程。"]
            elif days_since_signed is None:
                status = "blocked"
                reasons = ["缺少可信签收时间，无法判断是否仍在七天退货窗口内。"]
            elif not within_window:
                status = "not_eligible"
                reasons = ["订单签收已超过 7 天，不能按七天无理由进入退货申请。"]
            elif returnable is not True:
                status = "not_eligible" if returnable is False else "blocked"
                reasons = [
                    "商品明确不支持无理由退货。"
                    if returnable is False
                    else "缺少可信的商品可退属性，不能承诺可以退货。"
                ]
            else:
                status = "needs_clarification"
                reasons = ["请补充退货原因，例如七天无理由、质量问题、破损或错发。"]
        elif action_type == "cancel":
            status = "eligible_for_application" if not_shipped else "not_eligible"
            reasons = [
                (
                    "订单尚未发货，只能判断为可申请取消，不能直接修改订单状态。"
                    if not_shipped
                    else "订单已经发货或履约状态不明确，不能直接取消。"
                )
            ]
        elif action_type == "compensation":
            status = "manual_review_required"
            reasons = ["补偿金额和责任认定必须由人工复核，Agent 不具备审批权限。"]
        else:
            status = "needs_clarification"
            reasons = ["尚未明确要申请退款、退货、取消还是补偿。"]

        return HighRiskAssessment(
            action_type=action_type,
            order_id=order_id,
            eligibility_status=status,
            evidence_checklist=evidence,
            policy_basis=policy_basis,
            reasons=reasons,
            blocked_write_actions=BLOCKED_WRITE_ACTIONS,
        )
