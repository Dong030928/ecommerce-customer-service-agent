"""Lightweight and optional commercial reranking for retrieved candidates."""

from __future__ import annotations

from dataclasses import dataclass
import os

import httpx

from api.schemas import KnowledgeHit
from config.settings import (
    RERANK_INSTRUCTION_DEFAULT,
    RERANK_MODEL_DEFAULT,
    RERANK_TIMEOUT_SECONDS,
    api_key_is_missing,
    env_flag_enabled,
    load_project_env,
)
from rag.query_rewrite import normalize_query


@dataclass(frozen=True)
class RerankConfig:
    api_key: str
    base_url: str
    model: str
    instruction: str


@dataclass(frozen=True)
class RerankOutcome:
    hits: list[KnowledgeHit]
    mode: str
    model: str | None = None
    error: str | None = None


def explain_rerank_reasons(query: str, hit: KnowledgeHit) -> list[str]:
    """Explain deterministic boosts and penalties without exposing hidden reasoning."""

    query_text = normalize_query(query)
    status = hit.chunk.effective_status
    domain = str(hit.chunk.metadata.get("domain") or "")
    reasons: list[str] = []
    if status == "active" and "当前" in query_text:
        reasons.append("当前有效规则加权")
    if status in {"expired", "scheduled"}:
        reasons.append("非当前规则降权")
    if domain == "promotion" and all(term in query_text for term in ["优惠券", "叠加"]):
        reasons.append("优惠叠加条件匹配")
    if hit.chunk.chunk_id == "promotion-current-audio-offer" and "结算页" in query_text:
        reasons.append("结算页约束匹配")
    if domain != "promotion" and "活动" in query_text:
        reasons.append("非活动规则轻微降权")
    if hit.keyword_score is not None and hit.matched_keywords:
        reasons.append("精确关键词命中")
    if "keyword" in hit.retrieval_sources and any(
        source.endswith("vector") for source in hit.retrieval_sources
    ):
        reasons.append("向量与关键词双路命中")
    return reasons


def rerank_candidates_lightweight(
    query: str,
    candidates: list[KnowledgeHit],
) -> list[KnowledgeHit]:
    """Rerank candidates with transparent deterministic scoring signals."""

    reranked: list[KnowledgeHit] = []
    for hit in candidates:
        vector_score = hit.vector_score or 0.0
        keyword_score = hit.keyword_score or 0.0
        initial_score = max(vector_score, keyword_score, hit.score)
        # Reserve score headroom for reranking signals instead of saturating every
        # high-vector candidate at 1.0 before ordering can change.
        score = initial_score * 0.60
        reasons = explain_rerank_reasons(query, hit)
        if "当前有效规则加权" in reasons:
            score += 0.22
        if "非当前规则降权" in reasons:
            score -= 0.25
        if "优惠叠加条件匹配" in reasons:
            score += 0.12
        if "结算页约束匹配" in reasons:
            score += 0.08
        if "非活动规则轻微降权" in reasons:
            score -= 0.08
        if "精确关键词命中" in reasons:
            score += 0.12
        if "向量与关键词双路命中" in reasons:
            score += 0.08
        final_score = round(max(0.0, min(1.0, score)), 3)
        reranked.append(
            hit.model_copy(
                update={
                    "score": final_score,
                    "rerank_score": final_score,
                    "rerank_reasons": reasons or ["保留向量初始分"],
                }
            )
        )
    return sorted(reranked, key=lambda hit: hit.score, reverse=True)


def build_commercial_rerank_config() -> RerankConfig | None:
    """Resolve opt-in commercial reranker configuration."""

    load_project_env()
    if not env_flag_enabled("AGENT_RAG_RERANK_ENABLED"):
        return None
    api_key = os.getenv("AGENT_RAG_RERANK_API_KEY")
    if api_key_is_missing(api_key):
        api_key = os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        return None
    base_url = (
        os.getenv("AGENT_RAG_RERANK_BASE_URL")
        or os.getenv("AGENT_OPENAI_BASE_URL")
        or "https://api.siliconflow.cn/v1"
    ).rstrip("/")
    return RerankConfig(
        api_key=api_key or "",
        base_url=base_url,
        model=os.getenv("AGENT_RAG_RERANK_MODEL", RERANK_MODEL_DEFAULT),
        instruction=os.getenv(
            "AGENT_RAG_RERANK_INSTRUCTION",
            RERANK_INSTRUCTION_DEFAULT,
        ),
    )


def rerank_candidates_with_commercial_model(
    query: str,
    candidates: list[KnowledgeHit],
    config: RerankConfig,
    http_client: httpx.Client | None = None,
) -> list[KnowledgeHit]:
    """Call an OpenAI-compatible `/rerank` endpoint."""

    payload = {
        "model": config.model,
        "query": query,
        "documents": [hit.chunk.text for hit in candidates],
        "top_n": len(candidates),
        "return_documents": False,
    }
    if config.instruction:
        payload["instruction"] = config.instruction
    request_kwargs = {
        "headers": {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        "json": payload,
    }
    if http_client is not None:
        response = http_client.post(f"{config.base_url}/rerank", **request_kwargs)
    else:
        response = httpx.post(
            f"{config.base_url}/rerank",
            **request_kwargs,
            timeout=RERANK_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    results = response.json().get("results", [])
    ranked: list[KnowledgeHit] = []
    seen_indexes: set[int] = set()
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            continue
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        raw_score = item.get("relevance_score", item.get("score"))
        final_score = round(max(0.0, min(1.0, float(raw_score))), 3)
        ranked.append(
            candidates[index].model_copy(
                update={
                    "score": final_score,
                    "rerank_score": final_score,
                    "rerank_reasons": ["商业 reranker 精排"],
                }
            )
        )
    if len(ranked) != len(candidates):
        raise ValueError("Reranker 返回的有效结果数量与候选数量不一致。")
    return sorted(ranked, key=lambda hit: hit.score, reverse=True)


def rerank_candidates(
    query: str,
    candidates: list[KnowledgeHit],
    *,
    config: RerankConfig | None = None,
    http_client: httpx.Client | None = None,
) -> RerankOutcome:
    """Use the opted-in commercial reranker or a transparent local fallback."""

    if not candidates:
        return RerankOutcome(hits=[], mode="no_candidates")
    resolved_config = config or build_commercial_rerank_config()
    if resolved_config is None:
        return RerankOutcome(
            hits=rerank_candidates_lightweight(query, candidates),
            mode="lightweight",
        )
    try:
        ranked = rerank_candidates_with_commercial_model(
            query,
            candidates,
            resolved_config,
            http_client,
        )
        return RerankOutcome(
            hits=ranked,
            mode="commercial",
            model=resolved_config.model,
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return RerankOutcome(
            hits=rerank_candidates_lightweight(query, candidates),
            mode="commercial_fallback",
            model=resolved_config.model,
            error=exc.__class__.__name__,
        )
