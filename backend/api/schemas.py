"""Pydantic contracts for the public chat API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReasoningView = Literal["default", "off", "summary", "teaching"]
IntentSource = Literal["rules", "classifier", "rules_fallback"]
RouteSource = Literal["rules", "classifier", "rules_fallback"]
ExecutionRoute = Literal["general", "rag", "tool", "tool_rag", "workflow"]
Intent = Literal[
    "general_chat",
    "promotion_consult",
    "product_consult",
    "order_query",
    "refund_request",
    "refund_status_query",
    "complaint",
    "unknown",
]
ToolStatus = Literal["success", "error", "skipped"]
NextAction = Literal[
    "answer_user",
    "ask_clarification",
    "fallback_answer",
    "transfer_to_human",
]
RiskLevel = Literal["low", "medium", "high"]
HighRiskActionType = Literal[
    "refund",
    "return",
    "cancel",
    "compensation",
    "unknown",
]
ErrorCategory = Literal[
    "none",
    "timeout",
    "validation_error",
    "not_found",
    "forbidden",
    "business_error",
    "model_unavailable",
    "system_error",
    "high_risk_write_blocked",
]
HookType = Literal[
    "pre_tool_call",
    "post_tool_call",
    "on_error",
    "on_completion",
]
ClarificationSource = Literal["model", "backend_fallback", "backend_guard"]
RetrievalScene = Literal[
    "promotion",
    "after_sale",
    "shipping",
    "product",
    "order",
    "complaint",
    "unknown",
]


class ChatRequest(BaseModel):
    """Minimal request accepted by the `/chat` endpoint."""

    session_id: str = Field(..., description="当前对话会话 ID")
    # runtime_* values must come from trusted application state rather than user text.
    runtime_user_id: str = Field(..., description="可信调用方确认的用户 ID")
    runtime_nickname: str | None = Field(
        default=None, description="可信调用方确认的用户昵称"
    )
    runtime_member_level: str | None = Field(
        default=None, description="可信调用方确认的会员等级"
    )
    runtime_risk_level: str | None = Field(
        default=None, description="可信调用方确认的风险等级"
    )
    user_message: str = Field(..., description="用户输入的问题")
    reasoning_view: ReasoningView = "default"
    debug: bool = True
    runtime_context: dict[str, Any] | None = None


class IntentResult(BaseModel):
    """Stable, machine-readable classification result for downstream routing."""

    intent: Intent
    source: IntentSource
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)
    explanation: str


class RoutePlan(BaseModel):
    """Validated entry decision, not a hidden or long-form execution plan."""

    intent: Intent
    execution_route: ExecutionRoute
    needs_rag: bool
    needs_business_tools: bool
    rag_query: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: RouteSource
    intents: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    knowledge_domains: list[str] = Field(default_factory=list)
    has_realtime_fact: bool = False
    risk_level: RiskLevel = "low"
    requires_workflow: bool = False
    fallback_policy: str = "default"


class ToolCandidate(BaseModel):
    """Public-safe tool metadata considered for one route decision."""

    name: str
    domain: str
    allowed_in_light_path: bool
    risk_level: RiskLevel
    reason: str


class PlannerTrace(BaseModel):
    """Bounded planner trace that excludes hidden chain-of-thought."""

    source: RouteSource
    rule_confidence: float = Field(ge=0.0, le=1.0)
    model_consulted: bool = False
    safety_override: bool = False
    candidate_tools: list[ToolCandidate] = Field(default_factory=list)
    constrained_required_tools: list[str] = Field(default_factory=list)
    public_reason: str


class HighRiskAssessment(BaseModel):
    """Evidence-based eligibility result that never represents a write execution."""

    action_type: HighRiskActionType
    order_id: str | None = None
    eligibility_status: Literal[
        "eligible_for_application",
        "not_eligible",
        "needs_clarification",
        "manual_review_required",
        "blocked",
    ]
    risk_level: Literal["high"] = "high"
    needs_human_approval: bool = True
    evidence_checklist: list[str] = Field(default_factory=list)
    policy_basis: list["Citation"] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    blocked_write_actions: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """Public-safe contract shown to the model before it proposes a tool call."""

    name: str
    description: str
    required: list[str]
    parameters_schema: dict[str, str]
    read_only: bool = True
    risk_level: RiskLevel = "low"


class MCPToolDefinition(BaseModel):
    """MCP-style source definition with resource and prompt bindings."""

    name: str
    description: str
    required: list[str]
    parameters_schema: dict[str, str]
    read_only: bool = True
    risk_level: RiskLevel = "low"
    resource_uris: list[str] = Field(default_factory=list)
    prompt_ids: list[str] = Field(default_factory=list)

    def to_tool_spec(self) -> ToolSpec:
        """Adapt the catalog definition to the existing Tool Use contract."""

        return ToolSpec(
            name=self.name,
            description=self.description,
            required=list(self.required),
            parameters_schema=dict(self.parameters_schema),
            read_only=self.read_only,
            risk_level=self.risk_level,
        )


class MCPResource(BaseModel):
    """Stable boundary material bound to one or more catalog tools."""

    uri: str
    title: str
    content: str


class MCPPrompt(BaseModel):
    """Reusable public-safety wording bound to a catalog tool."""

    prompt_id: str
    title: str
    content: str


class MCPBindingSummary(BaseModel):
    """Public-safe summary of catalog bindings used by one response."""

    tool_source: Literal["mcp_catalog"] = "mcp_catalog"
    selected_tool: str | None = None
    selected_tools: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    boundary: str
    remote_server_connected: bool = False


class ToolAction(BaseModel):
    """Model-proposed tool name and arguments before backend validation."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str


class ToolResult(BaseModel):
    """Internal-only raw result; it must be compressed before model/public use."""

    tool_name: str
    status: ToolStatus
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_category: ErrorCategory = "none"
    attempts: int = Field(default=1, ge=1)
    source: str = "ecommerce_backend"


class ToolObservation(BaseModel):
    """Sanitized result returned by the controlled backend tool runtime."""

    tool_name: str
    status: ToolStatus
    summary: str
    facts: dict[str, Any] = Field(default_factory=dict)
    omitted_fields: list[str] = Field(default_factory=list)
    next_action: NextAction = "answer_user"
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_category: ErrorCategory = "none"
    attempts: int = Field(default=1, ge=1)
    source: str = "ecommerce_backend"


class ToolCallRecord(BaseModel):
    """Observable Action/Observation pair derived from the tool loop."""

    action: ToolAction
    observation: ToolObservation
    attempts: int = Field(default=1, ge=1)


class DegradationState(BaseModel):
    """Public-safe degradation decision for the current request."""

    degraded: bool = False
    error_category: ErrorCategory = "none"
    retry_count: int = Field(default=0, ge=0)
    fallback_used: bool = False
    reason: str | None = None


class HookEvent(BaseModel):
    """Public-safe lifecycle event; it never contains raw payloads or reasoning."""

    sequence: int = Field(ge=1)
    hook_type: HookType
    target_name: str
    action: str
    result: str
    reason: str
    safe_summary: dict[str, Any] = Field(default_factory=dict)
    redacted: bool = False
    pollution_detected: bool = False
    degraded: bool = False


class HookCompletion(BaseModel):
    """Bounded governance summary emitted once when a request finishes."""

    hook_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)
    touched_tools: list[str] = Field(default_factory=list)
    redacted_count: int = Field(ge=0)
    pollution_count: int = Field(ge=0)
    degraded_count: int = Field(ge=0)
    risk_hit_count: int = Field(ge=0)
    safe_summary: dict[str, Any] = Field(default_factory=dict)


class ClarificationCandidate(BaseModel):
    """Safe option that the user may choose without exposing private payloads."""

    value: str
    label: str
    hint: str


class ClarificationRequest(BaseModel):
    """Structured request returned when a tool target is missing or ambiguous."""

    clarification_field: str
    message: str
    candidates: list[ClarificationCandidate] = Field(default_factory=list)


class ClarificationPlan(BaseModel):
    """Model-draftable plan whose arguments are recomputed by the backend."""

    intent: Intent
    tool_name: str | None = None
    known_arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reason: str
    source: ClarificationSource
    model_name: str | None = None


class SourceDocument(BaseModel):
    """Repository-local Markdown source document."""

    source_path: str
    title: str
    metadata: dict[str, Any]
    body: str


class KnowledgeSection(BaseModel):
    """Metadata-bearing section parsed from a source document."""

    source_path: str
    document_title: str
    section_index: int = Field(ge=1)
    section: str
    chunk_id: str | None = None
    keywords: list[str] = Field(default_factory=list)
    effective_status: str = "active"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """Stable retrieval unit carrying its source and section metadata."""

    chunk_id: str
    document_title: str
    source_path: str
    section: str
    keywords: list[str] = Field(default_factory=list)
    effective_status: str = "active"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(BaseModel):
    """Knowledge chunk paired with its embedding vector."""

    chunk: KnowledgeChunk
    embedding: list[float]


class KnowledgeIndex(BaseModel):
    """Versioned in-process snapshot of the parsed knowledge corpus."""

    version: str
    fingerprint: str
    chunk_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    chunks_by_id: dict[str, KnowledgeChunk]
    inverted_index: dict[str, list[str]]


class KnowledgeHit(BaseModel):
    """Retrieved chunk with hybrid-retrieval and reranking evidence."""

    chunk: KnowledgeChunk
    score: float = Field(ge=0.0, le=1.0)
    vector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    keyword_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_sources: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    rerank_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank_reasons: list[str] = Field(default_factory=list)


class QueryRewrite(BaseModel):
    """Retrieval-only rewrite that preserves the user's original message."""

    original_query: str
    rewritten_query: str
    applied: bool
    added_terms: list[str] = Field(default_factory=list)
    reason: str


class RetrievalPlan(BaseModel):
    """Observable pre-retrieval routing and lexical-query plan."""

    original_query: str
    rewritten_query: str
    scene: RetrievalScene
    allowed_domains: list[str]
    keyword_terms: list[str]
    reason: str


class RetrievalCacheEntry(BaseModel):
    """Cached retrieval evidence; final answers are intentionally excluded."""

    index_version: str
    embedding_identity: str
    plan: RetrievalPlan
    original_vector_hits: list[KnowledgeHit]
    rewritten_vector_hits: list[KnowledgeHit]
    keyword_hits: list[KnowledgeHit]
    candidates: list[KnowledgeHit]


class Citation(BaseModel):
    """Public source citation derived from an actual retrieval hit."""

    citation_id: str
    source_title: str
    source_path: str
    section: str
    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    snippet: str


class RagQualityCase(BaseModel):
    """One fixed retrieval-quality regression case."""

    case_id: str
    question: str
    expected_chunk_ids: list[str]
    must_fallback: bool = False


class RagQualityCaseResult(BaseModel):
    """Recall, precision, and fallback result for one fixed case."""

    case_id: str
    retrieved_chunk_ids: list[str]
    expected_chunk_ids: list[str]
    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    fallback: bool
    passed: bool


class RagQualitySummary(BaseModel):
    """Lightweight aggregate over the fixed RAG quality set."""

    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    average_precision_at_k: float = Field(ge=0.0, le=1.0)
    results: list[RagQualityCaseResult]


class TokenUsage(BaseModel):
    """Normalized model-provider token usage."""

    prompt_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class CostSummary(BaseModel):
    """Public token and estimated cost observation for one request."""

    prompt_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    token_source: Literal["model_usage", "local_estimate"]
    usage_details: dict[str, Any] = Field(default_factory=dict)
    estimated_input_cost_cny: float = Field(ge=0)
    estimated_output_cost_cny: float = Field(ge=0)
    estimated_total_cost_cny: float = Field(ge=0)
    context_chars: int = Field(ge=0)
    pricing_note: str


class ChatResponse(BaseModel):
    """`/chat` 返回给调试后台的最小结构化响应。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    route_plan: RoutePlan | None = None
    planner_trace: PlannerTrace | None = None
    after_sale_assessment: HighRiskAssessment | None = None
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    hook_events: list[HookEvent] = Field(default_factory=list)
    hook_completion: HookCompletion | None = None
    mcp_context: MCPBindingSummary | None = None
    clarification: ClarificationRequest | None = None
    next_action: NextAction = "answer_user"
    risk_level: RiskLevel = "low"
    needs_human_approval: bool = False
    degraded: bool = False
    cost_summary: CostSummary
    reasoning_summary: list[str]
    session_state: dict[str, Any]


ChatRequest.model_rebuild(_types_namespace={"Any": Any, "ReasoningView": ReasoningView})
IntentResult.model_rebuild(
    _types_namespace={"Intent": Intent, "IntentSource": IntentSource}
)
RoutePlan.model_rebuild(
    _types_namespace={
        "ExecutionRoute": ExecutionRoute,
        "Intent": Intent,
        "RiskLevel": RiskLevel,
        "RouteSource": RouteSource,
    }
)
ToolCandidate.model_rebuild(_types_namespace={"RiskLevel": RiskLevel})
PlannerTrace.model_rebuild(
    _types_namespace={"RouteSource": RouteSource, "ToolCandidate": ToolCandidate}
)
HighRiskAssessment.model_rebuild(
    _types_namespace={
        "Citation": Citation,
        "HighRiskActionType": HighRiskActionType,
        "Literal": Literal,
    }
)
ToolSpec.model_rebuild(_types_namespace={"RiskLevel": RiskLevel})
MCPToolDefinition.model_rebuild(
    _types_namespace={"RiskLevel": RiskLevel, "ToolSpec": ToolSpec}
)
MCPResource.model_rebuild()
MCPPrompt.model_rebuild()
MCPBindingSummary.model_rebuild(_types_namespace={"Literal": Literal})
ToolAction.model_rebuild(_types_namespace={"Any": Any})
ToolResult.model_rebuild(
    _types_namespace={
        "Any": Any,
        "ErrorCategory": ErrorCategory,
        "ToolStatus": ToolStatus,
    }
)
ToolObservation.model_rebuild(
    _types_namespace={
        "Any": Any,
        "ErrorCategory": ErrorCategory,
        "NextAction": NextAction,
        "ToolStatus": ToolStatus,
    }
)
ToolCallRecord.model_rebuild(
    _types_namespace={"ToolAction": ToolAction, "ToolObservation": ToolObservation}
)
DegradationState.model_rebuild(
    _types_namespace={"ErrorCategory": ErrorCategory}
)
HookEvent.model_rebuild(
    _types_namespace={"Any": Any, "HookType": HookType}
)
HookCompletion.model_rebuild(_types_namespace={"Any": Any})
ClarificationCandidate.model_rebuild()
ClarificationRequest.model_rebuild(
    _types_namespace={"ClarificationCandidate": ClarificationCandidate}
)
ClarificationPlan.model_rebuild(
    _types_namespace={
        "Any": Any,
        "ClarificationSource": ClarificationSource,
        "Intent": Intent,
    }
)
KnowledgeChunk.model_rebuild(_types_namespace={"Any": Any})
VectorRecord.model_rebuild(_types_namespace={"KnowledgeChunk": KnowledgeChunk})
KnowledgeIndex.model_rebuild(_types_namespace={"KnowledgeChunk": KnowledgeChunk})
KnowledgeHit.model_rebuild(_types_namespace={"KnowledgeChunk": KnowledgeChunk})
QueryRewrite.model_rebuild()
RetrievalPlan.model_rebuild(_types_namespace={"RetrievalScene": RetrievalScene})
RetrievalCacheEntry.model_rebuild(
    _types_namespace={
        "KnowledgeHit": KnowledgeHit,
        "RetrievalPlan": RetrievalPlan,
    }
)
Citation.model_rebuild()
RagQualityCase.model_rebuild()
RagQualityCaseResult.model_rebuild()
RagQualitySummary.model_rebuild(
    _types_namespace={"RagQualityCaseResult": RagQualityCaseResult}
)
ChatResponse.model_rebuild(
    _types_namespace={
        "Any": Any,
        "CostSummary": CostSummary,
        "Citation": Citation,
        "Intent": Intent,
        "IntentResult": IntentResult,
        "HookCompletion": HookCompletion,
        "HookEvent": HookEvent,
        "HighRiskAssessment": HighRiskAssessment,
        "MCPBindingSummary": MCPBindingSummary,
        "NextAction": NextAction,
        "PlannerTrace": PlannerTrace,
        "RiskLevel": RiskLevel,
        "RoutePlan": RoutePlan,
        "ClarificationRequest": ClarificationRequest,
        "ToolCallRecord": ToolCallRecord,
    }
)
