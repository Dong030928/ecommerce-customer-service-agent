"""Create pending HITL requests without accepting chat text as a decision."""

from __future__ import annotations

from api.schemas import (
    AfterSaleWorkflowType,
    ApprovalRequest,
    HighRiskAssessment,
)


CHAT_APPROVAL_CLAIMS = [
    "审批通过",
    "主管同意",
    "我批准",
    "已经批准",
    "直接通过",
]


def is_chat_approval_claim(user_message: str) -> bool:
    """Reject untrusted natural language pretending to be an approval result."""

    return any(term in user_message for term in CHAT_APPROVAL_CLAIMS)


def build_approval_request(
    *,
    workflow_id: str,
    workflow_type: AfterSaleWorkflowType,
    assessment: HighRiskAssessment,
) -> ApprovalRequest:
    """Build a pending review contract; this does not approve or mutate an order."""

    action_label = "退款" if workflow_type == "unshipped_refund" else "退货"
    return ApprovalRequest(
        approval_id=f"appr-{workflow_id}",
        workflow_id=workflow_id,
        status="pending",
        required_role="after_sale_manager",
        submitted_by="authenticated_runtime_user",
        risk_summary=(
            f"订单 {assessment.order_id} 可发起{action_label}申请，"
            "但必须由售后主管通过受控审批通道处理。"
        ),
        decision_options=["approved", "rejected", "needs_more_info"],
        boundary=(
            "当前只创建待审批请求并暂停工作流；普通聊天不是审批通道，"
            "也尚未开放 /chat/resume。"
        ),
    )
