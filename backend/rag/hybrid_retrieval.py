"""Hybrid retrieval combining dual-query vectors and lexical evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from api.schemas import (
    Intent,
    KnowledgeHit,
    KnowledgeIndex,
    QueryRewrite,
    RetrievalCacheEntry,
    RetrievalPlan,
)
from config.settings import (
    HYBRID_CANDIDATE_K,
    KEYWORD_CANDIDATE_K,
    KEYWORD_SCORE_THRESHOLD,
)
from embeddings.client import EmbeddingClient, read_embedding_cache_identity
from rag.index_cache import (
    cache_entry_count,
    get_knowledge_index,
    get_retrieval_cache_entry,
    retrieval_cache_key,
    store_retrieval_cache_entry,
)
from rag.knowledge_base import (
    load_knowledge_chunks,
    query_asks_for_history,
    should_include_chunk_for_query,
)
from rag.planning import build_retrieval_plan, is_realtime_business_query
from rag.query_rewrite import normalize_query
from rag.retrieval import merge_candidates, retrieve_candidates


@dataclass(frozen=True)
class HybridRetrievalOutcome:
    plan: RetrievalPlan
    index: KnowledgeIndex
    original_vector_hits: list[KnowledgeHit]
    rewritten_vector_hits: list[KnowledgeHit]
    keyword_hits: list[KnowledgeHit]
    candidates: list[KnowledgeHit]
    cache: dict[str, Any]


def retrieval_cache_policy(plan: RetrievalPlan) -> dict[str, Any]:
    """Cache only stable retrieval candidates, never answers or realtime facts."""

    realtime = is_realtime_business_query(plan.original_query)
    cacheable = not realtime and plan.scene != "unknown"
    if realtime:
        reason = "实时订单、物流、库存或退款进度不能进入稳定知识检索缓存。"
    elif plan.scene == "unknown":
        reason = "未知场景不缓存，避免把跨领域弱召回长期复用。"
    else:
        reason = "只缓存稳定知识的 Hybrid RAG 候选，不缓存最终回答。"
    return {
        "cacheable": cacheable,
        "scope": "hybrid_candidates_only",
        "reason": reason,
    }


def keyword_match_score(term: str, searchable_text: str, keywords: list[str]) -> float:
    """Score exact lexical evidence; longer phrases carry more signal."""

    if term not in searchable_text and term not in keywords:
        return 0.0
    if term in keywords:
        return 1.5
    if len(term) >= 6:
        return 1.3
    if len(term) >= 4:
        return 1.1
    return 0.8


def retrieve_keyword_candidates(
    plan: RetrievalPlan,
    *,
    index: KnowledgeIndex | None = None,
    top_k: int = KEYWORD_CANDIDATE_K,
    threshold: float = KEYWORD_SCORE_THRESHOLD,
) -> list[KnowledgeHit]:
    """Retrieve exact long-tail and rule terms without calling an embedding API."""

    asks_for_history = query_asks_for_history(plan.original_query)
    hits: list[KnowledgeHit] = []
    source_chunks = (
        index.chunks_by_id.values() if index is not None else load_knowledge_chunks()
    )
    for chunk in source_chunks:
        if not should_include_chunk_for_query(chunk, asks_for_history):
            continue
        domain = str(chunk.metadata.get("domain") or "")
        if domain not in plan.allowed_domains:
            continue
        searchable = normalize_query(
            " ".join([chunk.document_title, chunk.section, *chunk.keywords, chunk.text])
        )
        normalized_keywords = [normalize_query(term) for term in chunk.keywords]
        matched = [
            term
            for term in plan.keyword_terms
            if keyword_match_score(
                normalize_query(term), searchable, normalized_keywords
            )
            > 0
        ]
        if not matched:
            continue
        evidence = sum(
            keyword_match_score(normalize_query(term), searchable, normalized_keywords)
            for term in matched
        )
        # One exact phrase is useful; multiple independent matches increase confidence.
        keyword_score = round(min(1.0, 0.42 + evidence * 0.12), 3)
        if keyword_score < threshold:
            continue
        hits.append(
            KnowledgeHit(
                chunk=chunk,
                score=keyword_score,
                keyword_score=keyword_score,
                retrieval_sources=["keyword"],
                matched_keywords=matched,
            )
        )
    return sorted(hits, key=lambda hit: hit.keyword_score or 0.0, reverse=True)[:top_k]


def retrieve_hybrid_candidates(
    rewrite: QueryRewrite,
    intent: Intent,
    *,
    embedding_client: EmbeddingClient | None = None,
) -> HybridRetrievalOutcome:
    """Run pre-retrieval planning and merge all three retrieval routes."""

    plan = build_retrieval_plan(rewrite, intent)
    index = get_knowledge_index()
    embedding_identity = read_embedding_cache_identity(embedding_client)
    identity_hash = hashlib.sha256(embedding_identity.encode("utf-8")).hexdigest()[:12]
    policy = retrieval_cache_policy(plan)
    cache_key = retrieval_cache_key(plan, index, embedding_identity)
    if policy["cacheable"]:
        cached = get_retrieval_cache_entry(cache_key)
        if cached is not None:
            return HybridRetrievalOutcome(
                plan=plan,
                index=index,
                original_vector_hits=cached.original_vector_hits,
                rewritten_vector_hits=cached.rewritten_vector_hits,
                keyword_hits=cached.keyword_hits,
                candidates=cached.candidates,
                cache={
                    **policy,
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "entry_count": cache_entry_count(),
                    "embedding_identity_hash": identity_hash,
                },
            )
    original_hits = retrieve_candidates(
        rewrite.original_query,
        embedding_client=embedding_client,
        index=index,
        allowed_domains=plan.allowed_domains,
        source="original_vector",
    )
    rewritten_hits = (
        retrieve_candidates(
            rewrite.rewritten_query,
            embedding_client=embedding_client,
            index=index,
            allowed_domains=plan.allowed_domains,
            source="rewritten_vector",
        )
        if rewrite.applied
        else []
    )
    keyword_hits = retrieve_keyword_candidates(plan, index=index)
    candidates = merge_candidates(
        original_hits,
        rewritten_hits,
        keyword_hits,
        top_k=HYBRID_CANDIDATE_K,
    )
    if policy["cacheable"]:
        store_retrieval_cache_entry(
            cache_key,
            RetrievalCacheEntry(
                index_version=index.version,
                embedding_identity=embedding_identity,
                plan=plan,
                original_vector_hits=original_hits,
                rewritten_vector_hits=rewritten_hits,
                keyword_hits=keyword_hits,
                candidates=candidates,
            ),
        )
    return HybridRetrievalOutcome(
        plan=plan,
        index=index,
        original_vector_hits=original_hits,
        rewritten_vector_hits=rewritten_hits,
        keyword_hits=keyword_hits,
        candidates=candidates,
        cache={
            **policy,
            "cache_hit": False,
            "cache_key": cache_key if policy["cacheable"] else None,
            "entry_count": cache_entry_count(),
            "embedding_identity_hash": identity_hash,
        },
    )
