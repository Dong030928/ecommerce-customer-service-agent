"""Assemble multi-source context with trust, visibility, and conflict rules."""

from __future__ import annotations

from api.schemas import (
    ChatRequest,
    Citation,
    ContextBuildReport,
    ContextItem,
    RuntimeContextView,
    SessionMemorySnapshot,
    ToolCallRecord,
    WorkflowSummary,
)
from context.runtime_context import user_claims_vip
from tools.planning import extract_order_id


TRUST_ORDER = {
    "trusted": 0,
    "verified": 1,
    "session": 2,
    "external": 3,
    "untrusted": 4,
}
APPROVAL_CLAIM_TERMS = (
    "客服说可以退",
    "上次说可以退",
    "已经批准",
    "主管同意",
    "审批通过",
)


class ContextBuilder:
    """Create one public-safe report without flattening source trust boundaries."""

    def __init__(self) -> None:
        self._selected: list[ContextItem] = []
        self._excluded: list[ContextItem] = []
        self._conflicts: list[str] = []

    def add(self, item: ContextItem) -> None:
        target = self._selected if item.allowed_for_model else self._excluded
        target.append(item)

    def build(
        self,
        *,
        request: ChatRequest,
        runtime: RuntimeContextView,
        memory: SessionMemorySnapshot,
        tool_calls: list[ToolCallRecord],
        citations: list[Citation],
        workflow: WorkflowSummary | None,
    ) -> ContextBuildReport:
        """Collect sources, apply deterministic precedence, and return a report."""

        self._add_user_message(request)
        self._add_runtime_context(runtime)
        self._add_memory(memory)
        self._add_tool_observations(tool_calls)
        self._add_rag_snippets(citations)
        self._add_workflow(workflow)
        self._resolve_conflicts(request, runtime, memory, workflow)
        ordered = sorted(
            self._selected,
            key=lambda item: TRUST_ORDER[item.trust_level],
        )
        return ContextBuildReport(
            selected_items=ordered,
            model_context=[
                f"[{item.source_type}/{item.trust_level}] {item.content}"
                for item in ordered
            ],
            conflict_resolutions=list(self._conflicts),
            excluded_items=list(self._excluded),
        )

    def _add_user_message(self, request: ChatRequest) -> None:
        self.add(
            ContextItem(
                item_id="user-message",
                source_type="user_message",
                trust_level="untrusted",
                content=request.user_message,
                facts={},
                conflict_group="user_claim",
                decision="只表示用户诉求，不能作为身份、审批或业务事实。",
            )
        )

    def _add_runtime_context(self, runtime: RuntimeContextView) -> None:
        model_facts = dict(runtime.trusted_for_model)
        self.add(
            ContextItem(
                item_id="runtime-context-model-view",
                source_type="runtime_context",
                trust_level="trusted",
                content=(
                    "可信登录态已提供；会员等级为 "
                    f"{model_facts.get('member_level', 'unknown')}，页面线索已最小化。"
                ),
                facts=model_facts,
                conflict_group="identity_and_page",
                decision="作为会员和页面上下文依据，优先于用户自述与 Session Memory。",
            )
        )
        self.add(
            ContextItem(
                item_id="runtime-context-system-only",
                source_type="runtime_context",
                trust_level="trusted",
                content="系统身份、风险等级和权限只用于后端校验。",
                facts={
                    "identity_source": runtime.system_only.get("identity_source"),
                    "user_id_present": runtime.system_only.get("user_id_present"),
                    "risk_level_present": runtime.system_only.get("risk_level")
                    not in {None, "unknown"},
                    "raw_user_id_exposed": False,
                },
                allowed_for_model=False,
                conflict_group="identity_and_permission",
                decision="系统专用授权信号不进入模型上下文，原始用户 ID 不进入报告。",
            )
        )

    def _add_memory(self, memory: SessionMemorySnapshot) -> None:
        facts = memory.model_dump(exclude={"excluded_items"})
        if not any(
            (
                memory.last_order_id,
                memory.last_product_name,
                memory.recent_intent,
                memory.low_risk_preferences,
            )
        ):
            return
        self.add(
            ContextItem(
                item_id="session-memory",
                source_type="session_memory",
                trust_level="session",
                content="当前会话存在可复用的最近实体、意图或低风险偏好。",
                facts=facts,
                conflict_group="session_reference",
                decision="仅辅助消歧，不能证明身份、覆盖页面上下文或修改 Workflow State。",
            )
        )

    def _add_tool_observations(self, tool_calls: list[ToolCallRecord]) -> None:
        for index, record in enumerate(tool_calls, start=1):
            observation = record.observation
            self.add(
                ContextItem(
                    item_id=f"tool-observation-{index}",
                    source_type="tool_observation",
                    trust_level="verified",
                    content=observation.summary,
                    facts=dict(observation.facts),
                    conflict_group="business_fact",
                    decision="采用经过白名单压缩的 Observation；原始 ToolResult 不进入上下文。",
                )
            )

    def _add_rag_snippets(self, citations: list[Citation]) -> None:
        for citation in citations:
            self.add(
                ContextItem(
                    item_id=f"rag-{citation.citation_id}",
                    source_type="rag_snippet",
                    trust_level="external",
                    content=citation.snippet,
                    facts={
                        "citation_id": citation.citation_id,
                        "chunk_id": citation.chunk_id,
                        "source_path": citation.source_path,
                        "score": citation.score,
                    },
                    conflict_group="policy_or_knowledge",
                    decision="作为有引用的稳定知识，但不能覆盖实时工具事实或流程状态。",
                )
            )

    def _add_workflow(self, workflow: WorkflowSummary | None) -> None:
        if workflow is None:
            return
        self.add(
            ContextItem(
                item_id="workflow-state",
                source_type="workflow_state",
                trust_level="verified",
                content=(
                    f"工作流状态 {workflow.status}，下一步 {workflow.pending_action}。"
                ),
                facts={
                    "workflow_type": workflow.workflow_type,
                    "status": workflow.status,
                    "current_node": workflow.current_node,
                    "pending_action": workflow.pending_action,
                    "approval_id": workflow.approval_id,
                    "has_resume_token": bool(workflow.resume_token),
                },
                conflict_group="workflow_state",
                decision="流程事实优先于历史说法；恢复令牌、幂等键和冻结字段不复制进模型上下文。",
            )
        )

    def _resolve_conflicts(
        self,
        request: ChatRequest,
        runtime: RuntimeContextView,
        memory: SessionMemorySnapshot,
        workflow: WorkflowSummary | None,
    ) -> None:
        if user_claims_vip(request.user_message) and runtime.trusted_for_model.get(
            "member_level"
        ) != "vip":
            self._conflicts.append(
                "member_level: 用户自称与 Runtime Context 冲突，采用可信 Runtime Context。"
            )

        explicit_order_id = extract_order_id(request.user_message)
        page_context = runtime.trusted_for_model.get("page_context") or {}
        page_order_id = page_context.get("related_order_no")
        if (
            explicit_order_id is None
            and page_order_id
            and memory.last_order_id
            and page_order_id != memory.last_order_id
        ):
            self._conflicts.append(
                "order_id: 页面 Runtime Context 与 Session Memory 冲突，采用页面上下文并重新校验。"
            )

        if workflow is not None and any(
            term in request.user_message for term in APPROVAL_CLAIM_TERMS
        ):
            self._conflicts.append(
                "workflow_state: 用户或历史客服说法不能覆盖 Workflow State，审批仍走受控恢复通道。"
            )
