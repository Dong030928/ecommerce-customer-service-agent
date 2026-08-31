"""Pydantic contracts for the public chat API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ReasoningView = Literal["default", "off", "summary", "teaching"]
IntentSource = Literal["rules", "classifier", "rules_fallback"]
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


class ToolSpec(BaseModel):
    """Public-safe contract shown to the model before it proposes a tool call."""

    name: str
    description: str
    required: list[str]
    parameters_schema: dict[str, str]


class ToolAction(BaseModel):
    """Model-proposed tool name and arguments before backend validation."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str


class ToolObservation(BaseModel):
    """Sanitized result returned by the controlled backend tool runtime."""

    tool_name: str
    status: ToolStatus
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    source: str = "ecommerce_backend"


class ToolCallRecord(BaseModel):
    """Observable Action/Observation pair derived from the tool loop."""

    action: ToolAction
    observation: ToolObservation


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
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    cost_summary: CostSummary
    reasoning_summary: list[str]
    session_state: dict[str, Any]


ChatRequest.model_rebuild(_types_namespace={"Any": Any, "ReasoningView": ReasoningView})
IntentResult.model_rebuild(
    _types_namespace={"Intent": Intent, "IntentSource": IntentSource}
)
ToolSpec.model_rebuild()
ToolAction.model_rebuild(_types_namespace={"Any": Any})
ToolObservation.model_rebuild(
    _types_namespace={"Any": Any, "ToolStatus": ToolStatus}
)
ToolCallRecord.model_rebuild(
    _types_namespace={"ToolAction": ToolAction, "ToolObservation": ToolObservation}
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
        "ToolCallRecord": ToolCallRecord,
    }
)
