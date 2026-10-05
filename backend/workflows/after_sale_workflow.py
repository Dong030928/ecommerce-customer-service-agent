"""LangGraph workflow that fixes the high-risk after-sale node order."""

from __future__ import annotations

from contextvars import ContextVar
import secrets
import sqlite3
from typing import Any, Callable, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from api.schemas import (
    AfterSaleWorkflowType,
    ApprovalRequest,
    ChatRequest,
    ChatResumeRequest,
    ChatResumeResponse,
    Citation,
    HighRiskActionType,
    HighRiskAssessment,
    IntentResult,
    ToolCallRecord,
    WorkflowStatus,
    WorkflowSummary,
)
from approvals.hitl import build_approval_request, is_chat_approval_claim
from hooks.manager import HookManager
from policies.after_sale_policy import (
    AfterSalePolicyService,
    clarification_assessment,
    detect_high_risk_action,
)
from state.approval_store import decision_fingerprint
from state.checkpoints import ApprovalSnapshot, WorkflowResumer, build_approval_snapshot
from state.native_checkpoint import WorkflowPersistence
from tools.planning import extract_order_id
from tools.runtime_context import contextual_order_id


PolicyRetriever = Callable[
    [ChatRequest, IntentResult],
    tuple[list[Citation], dict[str, Any]],
]


class AfterSaleWorkflowState(TypedDict, total=False):
    """Internal graph state; only WorkflowSummary crosses the public boundary."""

    schema_version: int
    request: dict[str, Any]
    intent_result: dict[str, Any]
    workflow_id: str
    workflow_type: AfterSaleWorkflowType
    action_type: HighRiskActionType
    order_id: str | None
    citations: list[dict[str, Any]]
    policy_state: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    assessment: dict[str, Any] | None
    approval: dict[str, Any] | None
    resume_seed: str
    resume_token: str | None
    idempotency_key: str | None
    frozen_fields: dict[str, Any]
    status: WorkflowStatus
    current_node: str
    pending_action: str
    node_history: list[str]
    answer: str
    used_langgraph: bool
    review: dict[str, Any]
    business_recheck: dict[str, Any]
    resume_response: dict[str, Any]


class AfterSaleWorkflow:
    """Run evidence and native approval nodes; never write to external business APIs."""

    def __init__(
        self,
        *,
        policy_service: AfterSalePolicyService,
        policy_retriever: PolicyRetriever,
        persistence: WorkflowPersistence | None = None,
    ) -> None:
        self._policy_service = policy_service
        self._policy_retriever = policy_retriever
        self.persistence = persistence or WorkflowPersistence.memory()
        self._store = self.persistence.approvals
        self._hook_context: ContextVar[HookManager | None] = ContextVar(
            "after_sale_request_hooks", default=None,
        )
        self._resumer = WorkflowResumer(
            store=self._store,
            policy_service=policy_service,
            agent_version="0.35.0",
            resume_graph=self._resume_graph,
            execution=self.persistence.execution,
        )
        self.graph = self._build_graph()

    @staticmethod
    def _request(state: AfterSaleWorkflowState) -> ChatRequest:
        return ChatRequest.model_validate(state["request"])

    def _hooks(self) -> HookManager:
        return self._hook_context.get() or HookManager()

    @staticmethod
    def _calls(state: AfterSaleWorkflowState) -> list[ToolCallRecord]:
        return [ToolCallRecord.model_validate(item) for item in state.get("tool_calls", [])]

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _hydrate(state: dict[str, Any]) -> dict[str, Any]:
        """Project primitive graph state back into existing typed Agent responses."""
        result = dict(state)
        for name, model in (("assessment", HighRiskAssessment), ("approval", ApprovalRequest)):
            if result.get(name) is not None:
                result[name] = model.model_validate(result[name])
        result["citations"] = [Citation.model_validate(item) for item in result.get("citations", [])]
        result["tool_calls"] = [ToolCallRecord.model_validate(item) for item in result.get("tool_calls", [])]
        if result.get("__interrupt__"):
            result["current_node"] = "human_review"
            result["node_history"] = [*result.get("node_history", []), "human_review"]
        return result

    def run(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        hooks: HookManager,
    ) -> AfterSaleWorkflowState:
        """Initialize one request-scoped graph state and execute it synchronously."""

        order_id = extract_order_id(request.user_message) or contextual_order_id(request)
        action_type = detect_high_risk_action(request.user_message)
        workflow_id = f"wf-{uuid4().hex}"
        if is_chat_approval_claim(request.user_message):
            return {
                "request": request,
                "intent_result": intent_result,
                "workflow_id": workflow_id,
                "workflow_type": "unknown",
                "action_type": action_type,
                "order_id": order_id,
                "citations": [],
                "policy_state": {
                    "status": "skipped_chat_approval_claim",
                    "mode": "hitl_guard",
                    "citation_count": 0,
                },
                "tool_calls": [],
                "assessment": HighRiskAssessment(
                    action_type=action_type,
                    order_id=order_id,
                    eligibility_status="blocked",
                    evidence_checklist=[],
                    reasons=["普通聊天消息不能作为售后主管的审批决策。"],
                    blocked_write_actions=[
                        "create_refund",
                        "approve_refund",
                        "approve_return",
                        "cancel_order",
                        "create_compensation",
                    ],
                ),
                "approval": None,
                "resume_token": None,
                "idempotency_key": None,
                "frozen_fields": {},
                "status": "blocked",
                "current_node": "reject_chat_approval_claim",
                "pending_action": "use_hitl_approval_channel",
                "node_history": ["reject_chat_approval_claim"],
                "answer": "普通聊天不能作为审批，审批结果必须来自受控 HITL 通道。",
                "used_langgraph": False,
            }
        initial_state: AfterSaleWorkflowState = {
            "schema_version": 1,
            "request": request.model_dump(mode="json", include={
                "session_id", "runtime_user_id", "user_message",
            }),
            "intent_result": intent_result.model_dump(mode="json"),
            "workflow_id": workflow_id,
            "workflow_type": "unknown",
            "action_type": action_type,
            "order_id": order_id,
            "citations": [],
            "policy_state": {
                "status": "not_started",
                "mode": "hybrid_rag_policy_evidence",
                "citation_count": 0,
            },
            "tool_calls": [],
            "assessment": None,
            "approval": None,
            "resume_token": None,
            "resume_seed": secrets.token_urlsafe(32),
            "idempotency_key": None,
            "frozen_fields": {},
            "status": "running",
            "current_node": "classify_after_sale_intent",
            "pending_action": "run_workflow",
            "node_history": [],
            "answer": "",
            "used_langgraph": True,
        }
        token = self._hook_context.set(hooks)
        try:
            with self.persistence.execution(workflow_id):
                result = self.graph.invoke(initial_state, self._config(workflow_id), durability="sync")
            return self._hydrate(result)
        except sqlite3.Error as exc:
            raise RuntimeError("售后工作流状态保存失败。") from exc
        finally:
            self._hook_context.reset(token)

    def resume(self, request: "ChatResumeRequest") -> "ChatResumeResponse":
        """Resume a paused approval only through the dedicated protocol."""

        return self._resumer.resume(request)

    def _resume_graph(self, request: ChatResumeRequest,
                      checkpoint: ApprovalSnapshot) -> ChatResumeResponse:
        config = self._config(checkpoint.thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values or snapshot.values.get("schema_version") != 1:
            return self._resumer._blocked(request, "原生图快照缺失或 Schema 版本不受支持。")
        saved, _, fingerprint = self._store.outcome(request.workflow_id)
        if snapshot.interrupts:
            if saved is not None and fingerprint == decision_fingerprint(request):
                replay = saved.model_copy(deep=True)
                replay.resume_result.idempotent_replay = True
                replay.session_state["resume_result"] = replay.resume_result.model_dump()
                return replay
            graph_input = Command(resume=request.model_dump(mode="json", exclude={"resume_token"}))
        elif snapshot.next and snapshot.values.get("review"):
            previous = self._review_request(snapshot.values)
            if decision_fingerprint(previous) != decision_fingerprint(request):
                return self._resumer._blocked(request, "上次审批恢复尚未完成，请以原请求重试。")
            graph_input = None  # Retry an already persisted graph decision, never approve anew.
        else:
            return self._resumer._blocked(request, "原生图没有可恢复的审批中断。")
        token = self._hook_context.set(HookManager())
        try:
            result = self.graph.invoke(graph_input, config, durability="sync")
        finally:
            self._hook_context.reset(token)
        payload = result.get("resume_response")
        if not payload:
            raise RuntimeError("售后工作流恢复后没有形成有效结果。")
        return ChatResumeResponse.model_validate(payload)

    def _build_graph(self):
        graph = StateGraph(AfterSaleWorkflowState)
        graph.add_node(
            "classify_after_sale_intent",
            self._classify_after_sale_intent,
        )
        graph.add_node("load_order", self._load_order)
        graph.add_node("load_logistics", self._load_logistics)
        graph.add_node("retrieve_policy", self._retrieve_policy)
        graph.add_node("check_eligibility", self._check_eligibility)
        graph.add_node("stop_before_submission", self._stop_before_submission)
        graph.add_node("prepare_approval", self._prepare_approval)
        graph.add_node("human_review", self._human_review)
        graph.add_node("resolve_review", self._resolve_review)
        graph.add_node("recheck_business", self._recheck_business)
        graph.add_node("submit_application", self._submit_application)
        graph.set_entry_point("classify_after_sale_intent")
        graph.add_conditional_edges(
            "classify_after_sale_intent",
            self._route_after_classify,
            {
                "load_order": "load_order",
                "stop_before_submission": "stop_before_submission",
            },
        )
        graph.add_conditional_edges(
            "load_order",
            self._route_after_order,
            {
                "load_logistics": "load_logistics",
                "stop_before_submission": "stop_before_submission",
            },
        )
        graph.add_edge("load_logistics", "retrieve_policy")
        graph.add_edge("retrieve_policy", "check_eligibility")
        graph.add_edge("check_eligibility", "stop_before_submission")
        graph.add_conditional_edges("stop_before_submission", lambda state:
            "prepare_approval" if state.get("approval") else END)
        graph.add_edge("prepare_approval", "human_review")
        graph.add_conditional_edges("human_review", lambda state:
            "recheck_business" if state["review"]["decision"] == "approved" else "resolve_review")
        graph.add_conditional_edges("resolve_review", lambda state:
            "human_review" if state["status"] == "paused" else END)
        graph.add_conditional_edges("recheck_business", lambda state:
            "submit_application" if state["business_recheck"]["passed"] else END)
        graph.add_edge("submit_application", END)
        return graph.compile(checkpointer=self.persistence.checkpointer)

    def _classify_after_sale_intent(
        self,
        state: AfterSaleWorkflowState,
    ) -> dict[str, Any]:
        workflow_types: dict[HighRiskActionType, AfterSaleWorkflowType] = {
            "refund": "unshipped_refund",
            "return": "received_return",
            "cancel": "order_cancellation",
            "compensation": "compensation_review",
            "unknown": "unknown",
        }
        return self._complete(
            state,
            "classify_after_sale_intent",
            {"workflow_type": workflow_types[state["action_type"]]},
        )

    def _load_order(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        record = self._policy_service.read_order(
            str(state["order_id"]),
            self._request(state),
            self._hooks(),
        )
        updates: dict[str, Any] = {
            "tool_calls": [*state["tool_calls"], record.model_dump(mode="json")],
        }
        if record.observation.status != "success":
            updates.update(
                {
                    "status": "blocked",
                    "pending_action": "transfer_to_human",
                    "answer": "订单事实或归属没有通过校验，售后流程不能继续。",
                }
            )
        return self._complete(state, "load_order", updates)

    def _load_logistics(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        record = self._policy_service.read_logistics(
            str(state["order_id"]),
            self._request(state),
            self._hooks(),
        )
        return self._complete(
            state,
            "load_logistics",
            {"tool_calls": [*state["tool_calls"], record.model_dump(mode="json")]},
        )

    def _retrieve_policy(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        citations, policy_state = self._policy_retriever(
            self._request(state),
            IntentResult.model_validate(state["intent_result"]),
        )
        return self._complete(
            state,
            "retrieve_policy",
            {"citations": [item.model_dump(mode="json") for item in citations], "policy_state": policy_state},
        )

    def _check_eligibility(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        calls = self._calls(state)
        order_call = next(
            (item for item in calls if item.action.tool_name == "get_order_status"),
            None,
        )
        logistics_call = next(
            (
                item
                for item in calls
                if item.action.tool_name == "get_order_logistics"
            ),
            None,
        )
        assessment = self._policy_service.assess_from_evidence(
            order_id=str(state["order_id"]),
            action_type=state["action_type"],
            policy_basis=[Citation.model_validate(item) for item in state["citations"]],
            order_call=order_call,
            logistics_call=logistics_call,
            user_message=self._request(state).user_message,
        )
        return self._complete(
            state,
            "check_eligibility",
            {"assessment": assessment.model_dump(mode="json")},
        )

    def _stop_before_submission(
        self,
        state: AfterSaleWorkflowState,
    ) -> dict[str, Any]:
        assessment = state.get("assessment")
        if assessment is not None:
            assessment = HighRiskAssessment.model_validate(assessment)
        if assessment is None:
            if not state.get("order_id") or state["workflow_type"] == "unknown":
                assessment = clarification_assessment(state["action_type"])
            else:
                order_call = next(
                    (
                        item
                        for item in self._calls(state)
                        if item.action.tool_name == "get_order_status"
                    ),
                    None,
                )
                assessment = self._policy_service.assess_from_evidence(
                    order_id=str(state["order_id"]),
                    action_type=state["action_type"],
                    policy_basis=[Citation.model_validate(item) for item in state["citations"]],
                    order_call=order_call,
                    logistics_call=None,
                    user_message=self._request(state).user_message,
                )
        status: WorkflowStatus = (
            "blocked"
            if assessment.eligibility_status in {"blocked", "needs_clarification"}
            else "completed"
        )
        approval: ApprovalRequest | None = None
        eligible = assessment.eligibility_status == "eligible_for_application"
        approvable_workflows = {"unshipped_refund", "received_return"}
        if eligible and state["workflow_type"] in approvable_workflows:
            approval = build_approval_request(
                workflow_id=state["workflow_id"],
                workflow_type=state["workflow_type"],
                assessment=assessment,
            )
            status = "paused"
            pending_action = "require_human_approval"
        else:
            pending_action = "explain_boundary"
        if eligible and state["workflow_type"] == "received_return":
            answer = "签收时间、商品可退属性、退货原因和政策依据均已核验；已创建待人工审批请求，但尚未批准或执行退货。"
        elif eligible and state["workflow_type"] == "unshipped_refund":
            answer = "订单已支付且未发货；已创建待人工审批请求，但尚未批准或执行退款。"
        else:
            answer = "售后工作流已完成资格判断，并在任何业务写入前停止。"
        return self._complete(
            state,
            "stop_before_submission",
            {
                "assessment": assessment.model_dump(mode="json"),
                "approval": approval.model_dump(mode="json") if approval else None,
                "status": status,
                "pending_action": pending_action,
                "answer": state.get("answer") or answer,
            },
        )

    def _prepare_approval(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        prepared = {**state, **self._complete(state, "prepare_approval")}
        snapshot = build_approval_snapshot(
            request=self._request(state), workflow=self.summary(prepared),
            approval=ApprovalRequest.model_validate(state["approval"]),
            assessment=HighRiskAssessment.model_validate(state["assessment"]),
            tool_calls=self._calls(state), resume_token=state["resume_seed"],
        )
        self._store.create(snapshot)
        return self._complete(state, "prepare_approval", {
            "resume_token": state["resume_seed"], "idempotency_key": snapshot.idempotency_key,
            "frozen_fields": snapshot.frozen_fields,
        })

    def _human_review(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        review = interrupt({
            "workflow_id": state["workflow_id"], "approval": state["approval"],
            "order_id": state["order_id"],
        })
        request = ChatResumeRequest.model_validate({**review, "resume_token": state["resume_token"]})
        if (request.workflow_id != state["workflow_id"]
                or request.session_id != self._request(state).session_id):
            raise RuntimeError("审批决策与原生图申请不匹配。")
        return self._complete(state, "human_review", {
            "review": request.model_dump(mode="json", exclude={"resume_token"}),
        })

    @staticmethod
    def _review_request(state: AfterSaleWorkflowState) -> ChatResumeRequest:
        return ChatResumeRequest.model_validate({**state["review"], "resume_token": state["resume_token"]})

    def _snapshot(self, state: AfterSaleWorkflowState) -> ApprovalSnapshot:
        snapshot = self._store.get(self._request(state).session_id, state["workflow_id"])
        if snapshot is None:
            raise RuntimeError("审批业务关联记录缺失。")
        return snapshot

    @staticmethod
    def _decorate_response(state: AfterSaleWorkflowState, node: str,
                           response: ChatResumeResponse) -> ChatResumeResponse:
        history = [*state.get("node_history", []), node]
        current = node
        if response.status == "paused":
            current = "human_review"
            history.append(current)
        response.workflow = response.workflow.model_copy(update={
            "current_node": current, "node_history": history,
        })
        response.session_state["workflow"] = response.workflow.model_dump()
        response.session_state["native_checkpoint"] = True
        return response

    def _response_updates(self, state: AfterSaleWorkflowState, node: str,
                          response: ChatResumeResponse) -> dict[str, Any]:
        return self._complete(state, node, {
            "resume_response": response.model_dump(mode="json"), "status": response.status,
            "answer": response.answer, "pending_action": response.workflow.pending_action,
            "approval": response.approval.model_dump(mode="json"),
        })

    def _resolve_review(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        request = self._review_request(state)
        response = self._resumer.resolve_without_submission(request, self._snapshot(state))
        response = self._decorate_response(state, "resolve_review", response)
        self._store.finish(request, response)
        return self._response_updates(state, "resolve_review", response)

    def _recheck_business(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        snapshot = self._snapshot(state)
        recheck = self._resumer._recheck(snapshot)
        if not recheck["passed"]:
            request = self._review_request(state)
            response = self._resumer.blocked_recheck(request, snapshot, recheck)
            response = self._decorate_response(state, "recheck_business", response)
            self._store.finish(request, response)
            return {**self._response_updates(state, "recheck_business", response),
                    "business_recheck": recheck}
        return self._complete(state, "recheck_business", {"business_recheck": recheck})

    def _submit_application(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        request, snapshot = self._review_request(state), self._snapshot(state)
        response = self._store.submit(request, snapshot.idempotency_key, lambda request_id, replay:
            self._decorate_response(state, "submit_application", self._resumer.approved_response(
                request, snapshot, state["business_recheck"], request_id, replay,
            )))
        return self._response_updates(state, "submit_application", response)

    @staticmethod
    def _route_after_classify(state: AfterSaleWorkflowState) -> str:
        if state.get("order_id") and state["workflow_type"] != "unknown":
            return "load_order"
        return "stop_before_submission"

    @staticmethod
    def _route_after_order(state: AfterSaleWorkflowState) -> str:
        latest = state["tool_calls"][-1] if state["tool_calls"] else None
        if latest is not None and ToolCallRecord.model_validate(latest).observation.status == "success":
            return "load_logistics"
        return "stop_before_submission"

    @staticmethod
    def _complete(
        state: AfterSaleWorkflowState,
        node: str,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            **(updates or {}),
            "current_node": node,
            "node_history": [*state.get("node_history", []), node],
        }

    @staticmethod
    def summary(state: AfterSaleWorkflowState) -> WorkflowSummary:
        """Compress internal objects into a stable frontend-facing summary."""

        boundary = (
            "审批结果必须来自受控 HITL 通道；普通聊天中的批准说法已被阻断。"
            if not state.get("used_langgraph", True)
            else (
                "资格通过后通过原生 interrupt 暂停；Resume 校验凭证与角色，"
                "恢复原图并复查业务事实、幂等记录模拟申请。"
            )
        )
        return WorkflowSummary(
            workflow_id=state["workflow_id"],
            workflow_type=state["workflow_type"],
            status=state["status"],
            current_node=state["current_node"],
            pending_action=state["pending_action"],
            node_history=state["node_history"],
            used_langgraph=state.get("used_langgraph", True),
            boundary=boundary,
            approval_id=(
                ApprovalRequest.model_validate(state["approval"]).approval_id if state.get("approval") else None
            ),
            resume_token=state.get("resume_token"),
            idempotency_key=state.get("idempotency_key"),
            frozen_fields=state.get("frozen_fields", {}),
        )
