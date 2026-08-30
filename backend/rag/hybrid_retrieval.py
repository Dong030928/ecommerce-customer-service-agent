"""Hybrid retrieval combining dual-query vectors and lexical evidence."""

from __future__ import annotations

from dataclasses import dataclass

from api.schemas import Intent, KnowledgeHit, QueryRewrite, RetrievalPlan
from config.settings import (
    HYBRID_CANDIDATE_K,
    KEYWORD_CANDIDATE_K,
    KEYWORD_SCORE_THRESHOLD,
)
from embeddings.client import EmbeddingClient
from rag.knowledge_base import (
    load_knowledge_chunks,
    query_asks_for_history,
    should_include_chunk_for_query,
)
from rag.planning import build_retrieval_plan
from rag.query_rewrite import normalize_query
from rag.retrieval import merge_candidates, retrieve_candidates


@dataclass(frozen=True)
class HybridRetrievalOutcome:
    plan: RetrievalPlan
    original_vector_hits: list[KnowledgeHit]
    rewritten_vector_hits: list[KnowledgeHit]
    keyword_hits: list[KnowledgeHit]
    candidates: list[KnowledgeHit]


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
    top_k: int = KEYWORD_CANDIDATE_K,
    threshold: float = KEYWORD_SCORE_THRESHOLD,
) -> list[KnowledgeHit]:
    """Retrieve exact long-tail and rule terms without calling an embedding API."""

    asks_for_history = query_asks_for_history(plan.original_query)
    hits: list[KnowledgeHit] = []
    for chunk in load_knowledge_chunks():
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
    original_hits = retrieve_candidates(
        rewrite.original_query,
        embedding_client=embedding_client,
        allowed_domains=plan.allowed_domains,
        source="original_vector",
    )
    rewritten_hits = (
        retrieve_candidates(
            rewrite.rewritten_query,
            embedding_client=embedding_client,
            allowed_domains=plan.allowed_domains,
            source="rewritten_vector",
        )
        if rewrite.applied
        else []
    )
    keyword_hits = retrieve_keyword_candidates(plan)
    candidates = merge_candidates(
        original_hits,
        rewritten_hits,
        keyword_hits,
        top_k=HYBRID_CANDIDATE_K,
    )
    return HybridRetrievalOutcome(
        plan=plan,
        original_vector_hits=original_hits,
        rewritten_vector_hits=rewritten_hits,
        keyword_hits=keyword_hits,
        candidates=candidates,
    )
