"""LangGraph workflow that fixes the high-risk after-sale node order."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from api.schemas import (
    AfterSaleWorkflowType,
    ChatRequest,
    Citation,
    HighRiskActionType,
    HighRiskAssessment,
    IntentResult,
    ToolCallRecord,
    WorkflowStatus,
    WorkflowSummary,
)
from hooks.manager import HookManager
from policies.after_sale_policy import (
    AfterSalePolicyService,
    clarification_assessment,
    detect_high_risk_action,
)
from tools.planning import extract_order_id


PolicyRetriever = Callable[
    [ChatRequest, IntentResult],
    tuple[list[Citation], dict[str, Any]],
]


class AfterSaleWorkflowState(TypedDict, total=False):
    """Internal graph state; only WorkflowSummary crosses the public boundary."""

    request: ChatRequest
    intent_result: IntentResult
    hooks: HookManager
    workflow_id: str
    workflow_type: AfterSaleWorkflowType
    action_type: HighRiskActionType
    order_id: str | None
    citations: list[Citation]
    policy_state: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    assessment: HighRiskAssessment | None
    status: WorkflowStatus
    current_node: str
    pending_action: str
    node_history: list[str]
    answer: str


class AfterSaleWorkflow:
    """Run a fixed evidence workflow and stop before every write operation."""

    def __init__(
        self,
        *,
        policy_service: AfterSalePolicyService,
        policy_retriever: PolicyRetriever,
    ) -> None:
        self._policy_service = policy_service
        self._policy_retriever = policy_retriever
        self.graph = self._build_graph()

    def run(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        hooks: HookManager,
    ) -> AfterSaleWorkflowState:
        """Initialize one request-scoped graph state and execute it synchronously."""

        order_id = extract_order_id(request.user_message)
        initial_state: AfterSaleWorkflowState = {
            "request": request,
            "intent_result": intent_result,
            "hooks": hooks,
            "workflow_id": f"wf-{request.session_id}-{order_id or 'missing'}",
            "workflow_type": "unknown",
            "action_type": detect_high_risk_action(request.user_message),
            "order_id": order_id,
            "citations": [],
            "policy_state": {
                "status": "not_started",
                "mode": "hybrid_rag_policy_evidence",
                "citation_count": 0,
            },
            "tool_calls": [],
            "assessment": None,
            "status": "running",
            "current_node": "classify_after_sale_intent",
            "pending_action": "run_workflow",
            "node_history": [],
            "answer": "",
        }
        return self.graph.invoke(initial_state)

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
        graph.add_edge("stop_before_submission", END)
        return graph.compile()

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
            state["request"],
            state["hooks"],
        )
        updates: dict[str, Any] = {
            "tool_calls": [*state["tool_calls"], record],
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
            state["request"],
            state["hooks"],
        )
        return self._complete(
            state,
            "load_logistics",
            {"tool_calls": [*state["tool_calls"], record]},
        )

    def _retrieve_policy(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        citations, policy_state = self._policy_retriever(
            state["request"],
            state["intent_result"],
        )
        return self._complete(
            state,
            "retrieve_policy",
            {"citations": citations, "policy_state": policy_state},
        )

    def _check_eligibility(self, state: AfterSaleWorkflowState) -> dict[str, Any]:
        calls = state["tool_calls"]
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
            policy_basis=state["citations"],
            order_call=order_call,
            logistics_call=logistics_call,
            user_message=state["request"].user_message,
        )
        return self._complete(
            state,
            "check_eligibility",
            {"assessment": assessment},
        )

    def _stop_before_submission(
        self,
        state: AfterSaleWorkflowState,
    ) -> dict[str, Any]:
        assessment = state.get("assessment")
        if assessment is None:
            if not state.get("order_id") or state["workflow_type"] == "unknown":
                assessment = clarification_assessment(state["action_type"])
            else:
                order_call = next(
                    (
                        item
                        for item in state["tool_calls"]
                        if item.action.tool_name == "get_order_status"
                    ),
                    None,
                )
                assessment = self._policy_service.assess_from_evidence(
                    order_id=str(state["order_id"]),
                    action_type=state["action_type"],
                    policy_basis=state["citations"],
                    order_call=order_call,
                    logistics_call=None,
                    user_message=state["request"].user_message,
                )
        status: WorkflowStatus = (
            "blocked"
            if assessment.eligibility_status in {"blocked", "needs_clarification"}
            else "completed"
        )
        eligible = assessment.eligibility_status == "eligible_for_application"
        pending_actions = {
            "unshipped_refund": "prepare_refund_application",
            "received_return": "prepare_return_application",
        }
        pending_action = (
            pending_actions.get(state["workflow_type"], "explain_boundary")
            if eligible
            else "explain_boundary"
        )
        if eligible and state["workflow_type"] == "received_return":
            answer = "签收时间、商品可退属性、退货原因和政策依据均已核验；当前只可准备退货申请，尚未提交或批准。"
        elif eligible and state["workflow_type"] == "unshipped_refund":
            answer = "订单已支付且未发货；当前只可准备退款申请，尚未提交、退款或完成人工审批。"
        else:
            answer = "售后工作流已完成资格判断，并在任何业务写入前停止。"
        return self._complete(
            state,
            "stop_before_submission",
            {
                "assessment": assessment,
                "status": status,
                "pending_action": pending_action,
                "answer": state.get("answer") or answer,
            },
        )

    @staticmethod
    def _route_after_classify(state: AfterSaleWorkflowState) -> str:
        if state.get("order_id") and state["workflow_type"] != "unknown":
            return "load_order"
        return "stop_before_submission"

    @staticmethod
    def _route_after_order(state: AfterSaleWorkflowState) -> str:
        latest = state["tool_calls"][-1] if state["tool_calls"] else None
        if latest is not None and latest.observation.status == "success":
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

        return WorkflowSummary(
            workflow_id=state["workflow_id"],
            workflow_type=state["workflow_type"],
            status=state["status"],
            current_node=state["current_node"],
            pending_action=state["pending_action"],
            node_history=state["node_history"],
            used_langgraph=True,
            boundary=(
                "StateGraph 固定售后节点顺序；当前不提交申请、不执行审批、"
                "不持久化 checkpoint，也不返回 resume_token。"
            ),
        )
