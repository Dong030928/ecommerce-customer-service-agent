"""High-risk after-sale eligibility checks built from read-only evidence."""

from __future__ import annotations

from dataclasses import dataclass

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

        order_call = self._read("get_order_status", order_id, request, hooks)
        logistics_call = self._read("get_order_logistics", order_id, request, hooks)
        tool_calls = [order_call, logistics_call]
        failed = [
            record.observation
            for record in tool_calls
            if record.observation.status != "success"
        ]
        evidence = ["订单状态", "支付状态", "物流状态"]
        if policy_basis:
            evidence.append("售后政策引用")
        if failed:
            return AfterSaleBoundaryResult(
                assessment=HighRiskAssessment(
                    action_type=action_type,
                    order_id=order_id,
                    eligibility_status="blocked",
                    evidence_checklist=evidence,
                    policy_basis=policy_basis,
                    reasons=["未取得完整可信的订单或物流事实，不能判断售后资格。"],
                    blocked_write_actions=BLOCKED_WRITE_ACTIONS,
                ),
                tool_calls=tool_calls,
            )

        order_facts = order_call.observation.facts
        logistics_facts = logistics_call.observation.facts
        order_status = _normalized(order_facts.get("order_status"))
        payment_status = _normalized(order_facts.get("payment_status"))
        logistics_status = _normalized(logistics_facts.get("logistics_status"))
        not_shipped = logistics_status in {
            "",
            "NOT_SHIPPED",
            "PENDING_SHIPMENT",
            "PENDING",
            "待发货",
        } and order_status not in {"SHIPPED", "DELIVERED", "COMPLETED", "已发货", "已完成"}
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
            status = "eligible_for_application" if signed else "not_eligible"
            reasons = [
                (
                    "订单已签收，可进入退货申请资格复核，但不得自动批准。"
                    if signed
                    else "订单尚未形成可信签收事实，不能判断为可发起签收后退货。"
                )
            ]
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

        return AfterSaleBoundaryResult(
            assessment=HighRiskAssessment(
                action_type=action_type,
                order_id=order_id,
                eligibility_status=status,
                evidence_checklist=evidence,
                policy_basis=policy_basis,
                reasons=reasons,
                blocked_write_actions=BLOCKED_WRITE_ACTIONS,
            ),
            tool_calls=tool_calls,
        )
