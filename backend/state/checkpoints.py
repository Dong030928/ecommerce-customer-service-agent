"""In-process checkpoint, resume validation, business recheck, and idempotency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
from threading import RLock
from typing import Any

from api.schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResumeRequest,
    ChatResumeResponse,
    HighRiskAssessment,
    ResumeResult,
    ToolCallRecord,
    WorkflowSummary,
)
from hooks.manager import HookManager
from policies.after_sale_policy import AfterSalePolicyService


RECHECK_FACT_KEYS = (
    "order_status",
    "payment_status",
    "total_amount",
    "fulfillment_status",
    "delivered_at",
    "returnable",
    "logistics_status",
)


@dataclass
class WorkflowCheckpoint:
    session_id: str
    workflow_id: str
    requester_id: str
    workflow: WorkflowSummary
    approval: ApprovalRequest
    assessment: HighRiskAssessment
    resume_token: str
    idempotency_key: str
    frozen_fields: dict[str, Any]


def _record(
    tool_calls: list[ToolCallRecord],
    tool_name: str,
) -> ToolCallRecord | None:
    return next(
        (item for item in tool_calls if item.action.tool_name == tool_name),
        None,
    )


def _safe_business_facts(tool_calls: list[ToolCallRecord]) -> dict[str, Any]:
    order_call = _record(tool_calls, "get_order_status")
    logistics_call = _record(tool_calls, "get_order_logistics")
    order_facts = order_call.observation.facts if order_call else {}
    logistics_facts = logistics_call.observation.facts if logistics_call else {}
    return {
        "order_status": order_facts.get("order_status"),
        "payment_status": order_facts.get("payment_status"),
        "total_amount": order_facts.get("total_amount"),
        "fulfillment_status": order_facts.get("fulfillment_status"),
        "delivered_at": (
            logistics_facts.get("delivered_at")
            or order_facts.get("delivered_at")
        ),
        "returnable": order_facts.get("returnable"),
        "logistics_status": logistics_facts.get("logistics_status"),
    }


def _requester_fingerprint(requester_id: str, salt: str) -> str:
    return hmac.new(
        salt.encode("utf-8"),
        requester_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]


class CheckpointStore:
    """Process-local lesson store; a durable backend can replace this interface."""

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], WorkflowCheckpoint] = {}
        self._submitted_actions: dict[str, str] = {}
        self._lock = RLock()

    def create(
        self,
        *,
        request: ChatRequest,
        workflow: WorkflowSummary,
        approval: ApprovalRequest,
        assessment: HighRiskAssessment,
        tool_calls: list[ToolCallRecord],
    ) -> WorkflowCheckpoint:
        resume_token = secrets.token_urlsafe(32)
        key_material = (
            f"{workflow.workflow_id}:{assessment.action_type}:"
            f"{assessment.order_id or 'missing'}"
        )
        idempotency_key = "hitl-" + hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest()[:24]
        frozen_fields = {
            "workflow_type": workflow.workflow_type,
            "order_id": assessment.order_id,
            "requester_fingerprint": _requester_fingerprint(
                request.runtime_user_id,
                resume_token,
            ),
            "eligibility_status": assessment.eligibility_status,
            "policy_citation_ids": [
                citation.citation_id for citation in assessment.policy_basis
            ],
            **_safe_business_facts(tool_calls),
        }
        checkpoint = WorkflowCheckpoint(
            session_id=request.session_id,
            workflow_id=workflow.workflow_id,
            requester_id=request.runtime_user_id,
            workflow=workflow.model_copy(
                update={
                    "resume_token": resume_token,
                    "idempotency_key": idempotency_key,
                    "frozen_fields": frozen_fields,
                }
            ),
            approval=approval,
            assessment=assessment,
            resume_token=resume_token,
            idempotency_key=idempotency_key,
            frozen_fields=frozen_fields,
        )
        with self._lock:
            self._checkpoints[(request.session_id, workflow.workflow_id)] = checkpoint
        return checkpoint

    def get(self, session_id: str, workflow_id: str) -> WorkflowCheckpoint | None:
        with self._lock:
            return self._checkpoints.get((session_id, workflow_id))

    def submitted_request(self, idempotency_key: str) -> str | None:
        with self._lock:
            return self._submitted_actions.get(idempotency_key)

    def record_submission(self, idempotency_key: str) -> tuple[str, bool]:
        with self._lock:
            existing = self._submitted_actions.get(idempotency_key)
            if existing is not None:
                return existing, True
            request_id = "asr-" + hashlib.sha256(
                idempotency_key.encode("utf-8")
            ).hexdigest()[:12]
            self._submitted_actions[idempotency_key] = request_id
            return request_id, False


class WorkflowResumer:
    """Resume only through a validated checkpoint and fresh read-only facts."""

    def __init__(
        self,
        *,
        store: CheckpointStore,
        policy_service: AfterSalePolicyService,
        agent_version: str,
    ) -> None:
        self._store = store
        self._policy_service = policy_service
        self._agent_version = agent_version

    def _response(
        self,
        request: ChatResumeRequest,
        *,
        status: str,
        answer: str,
        result: ResumeResult,
        checkpoint: WorkflowCheckpoint | None = None,
        workflow: WorkflowSummary | None = None,
        approval: ApprovalRequest | None = None,
        business_recheck: dict[str, Any] | None = None,
    ) -> ChatResumeResponse:
        current_workflow = workflow or (
            checkpoint.workflow if checkpoint is not None else None
        )
        current_approval = approval or (
            checkpoint.approval if checkpoint is not None else None
        )
        recheck = business_recheck or {}
        return ChatResumeResponse(
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            status=status,
            answer=answer,
            workflow=current_workflow,
            approval=current_approval,
            resume_result=result,
            business_recheck=recheck,
            session_state={
                "agent_version": self._agent_version,
                "workflow": (
                    current_workflow.model_dump() if current_workflow else None
                ),
                "approval": (
                    current_approval.model_dump() if current_approval else None
                ),
                "resume_result": result.model_dump(),
                "business_recheck": recheck,
                "external_business_write_executed": False,
            },
        )

    def _blocked(
        self,
        request: ChatResumeRequest,
        reason: str,
        *,
        checkpoint: WorkflowCheckpoint | None = None,
        business_recheck: dict[str, Any] | None = None,
    ) -> ChatResumeResponse:
        return self._response(
            request,
            status="blocked",
            answer=reason,
            checkpoint=checkpoint,
            result=ResumeResult(
                decision=request.decision,
                accepted=False,
                reason=reason,
            ),
            business_recheck=business_recheck,
        )

    def _recheck(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> dict[str, Any]:
        order_id = str(checkpoint.frozen_fields.get("order_id") or "")
        request = ChatRequest(
            session_id=checkpoint.session_id,
            runtime_user_id=checkpoint.requester_id,
            user_message="HITL resume business fact recheck",
        )
        hooks = HookManager()
        order_call = self._policy_service.read_order(order_id, request, hooks)
        logistics_call = self._policy_service.read_logistics(
            order_id,
            request,
            hooks,
        )
        calls = [order_call, logistics_call]
        if any(call.observation.status != "success" for call in calls):
            return {
                "passed": False,
                "reason": "恢复时未取得完整可信的订单或物流事实。",
                "mismatches": {},
            }
        current = _safe_business_facts(calls)
        mismatches = {
            key: {
                "frozen": checkpoint.frozen_fields.get(key),
                "current": current.get(key),
            }
            for key in RECHECK_FACT_KEYS
            if checkpoint.frozen_fields.get(key) != current.get(key)
        }
        return {
            "passed": not mismatches,
            "reason": (
                "业务事实二次校验通过。"
                if not mismatches
                else "业务事实已变化，不能沿用旧审批结果。"
            ),
            "mismatches": mismatches,
            "checked_fields": list(RECHECK_FACT_KEYS),
        }

    def resume(self, request: ChatResumeRequest) -> ChatResumeResponse:
        checkpoint = self._store.get(request.session_id, request.workflow_id)
        if checkpoint is None:
            return self._blocked(request, "没有找到匹配会话与工作流的 checkpoint。")
        if not hmac.compare_digest(request.resume_token, checkpoint.resume_token):
            return self._blocked(
                request,
                "resume_token 不匹配，不能恢复这个审批流程。",
                checkpoint=checkpoint,
            )
        if not request.reviewer_id.strip():
            return self._blocked(
                request,
                "缺少可信审批人标识。",
                checkpoint=checkpoint,
            )
        if request.reviewer_role != checkpoint.approval.required_role:
            return self._blocked(
                request,
                "只有售后主管角色可以恢复高风险审批。",
                checkpoint=checkpoint,
            )

        if request.decision == "rejected":
            workflow = checkpoint.workflow.model_copy(
                update={"status": "rejected", "pending_action": "notify_user"}
            )
            approval = checkpoint.approval.model_copy(update={"status": "rejected"})
            return self._response(
                request,
                status="rejected",
                answer="售后主管已拒绝该申请，系统不会提交业务动作。",
                workflow=workflow,
                approval=approval,
                result=ResumeResult(
                    decision="rejected",
                    accepted=True,
                    reason="人工审批拒绝，未记录售后申请。",
                ),
                business_recheck={
                    "passed": True,
                    "reason": "rejected_without_submission",
                },
            )

        if request.decision == "needs_more_info":
            workflow = checkpoint.workflow.model_copy(
                update={"status": "paused", "pending_action": "ask_user"}
            )
            approval = checkpoint.approval.model_copy(
                update={"status": "needs_more_info"}
            )
            return self._response(
                request,
                status="paused",
                answer="售后主管要求补充信息，当前不会记录业务申请。",
                workflow=workflow,
                approval=approval,
                result=ResumeResult(
                    decision="needs_more_info",
                    accepted=True,
                    reason="流程继续暂停，等待补充信息。",
                ),
                business_recheck={
                    "passed": True,
                    "reason": "needs_more_info_without_submission",
                },
            )

        recheck = self._recheck(checkpoint)
        if not recheck["passed"]:
            workflow = checkpoint.workflow.model_copy(
                update={"status": "blocked", "pending_action": "transfer_to_human"}
            )
            return self._response(
                request,
                status="blocked",
                answer="恢复时业务事实已经变化，请人工重新核验。",
                workflow=workflow,
                approval=checkpoint.approval,
                result=ResumeResult(
                    decision="approved",
                    accepted=False,
                    reason=str(recheck["reason"]),
                ),
                business_recheck=recheck,
            )

        request_id, replay = self._store.record_submission(
            checkpoint.idempotency_key
        )
        workflow = checkpoint.workflow.model_copy(
            update={"status": "completed", "pending_action": "notify_user"}
        )
        approval = checkpoint.approval.model_copy(update={"status": "approved"})
        reason = (
            "重复恢复命中同一幂等键，未重复记录。"
            if replay
            else "人工审批通过，已幂等记录售后申请。"
        )
        return self._response(
            request,
            status="completed",
            answer=(
                f"审批已通过，售后申请记录为 {request_id}。"
                "这不代表外部支付退款已经完成。"
            ),
            workflow=workflow,
            approval=approval,
            result=ResumeResult(
                decision="approved",
                accepted=True,
                idempotent_replay=replay,
                request_id=request_id,
                reason=reason,
            ),
            business_recheck=recheck,
        )
