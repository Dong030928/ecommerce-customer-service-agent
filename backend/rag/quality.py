"""Fixed-set RAG quality metrics and low-confidence decisions."""

from __future__ import annotations

import json

from api.schemas import (
    KnowledgeHit,
    RagQualityCase,
    RagQualityCaseResult,
    RagQualitySummary,
)
from config.settings import LOW_CONFIDENCE_THRESHOLD, QUALITY_CASES_PATH
from embeddings.client import EmbeddingClient
from rag.reranker import rerank_candidates_lightweight
from rag.retrieval import retrieve_candidates


def load_quality_cases() -> list[RagQualityCase]:
    """Load the repository's fixed RAG quality cases."""

    with QUALITY_CASES_PATH.open(encoding="utf-8") as file:
        return [RagQualityCase.model_validate(item) for item in json.load(file)]


def is_low_confidence(hits: list[KnowledgeHit]) -> bool:
    """Decide whether the best candidate is too weak to support an answer."""

    return not hits or hits[0].score < LOW_CONFIDENCE_THRESHOLD


def evaluate_quality_case(
    case: RagQualityCase,
    embedding_client: EmbeddingClient | None = None,
) -> RagQualityCaseResult:
    """Evaluate recall@k, precision@k, and fallback behavior for one case."""

    candidates = retrieve_candidates(case.question, embedding_client=embedding_client)
    hits = rerank_candidates_lightweight(case.question, candidates)
    fallback = is_low_confidence(hits)
    retrieved_ids = [] if fallback else [hit.chunk.chunk_id for hit in hits]
    expected = set(case.expected_chunk_ids)
    retrieved = set(retrieved_ids)

    if case.must_fallback:
        return RagQualityCaseResult(
            case_id=case.case_id,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=case.expected_chunk_ids,
            recall_at_k=1.0 if fallback else 0.0,
            precision_at_k=1.0 if fallback else 0.0,
            fallback=fallback,
            passed=fallback and not retrieved_ids,
        )

    matched = expected & retrieved
    recall = len(matched) / max(len(expected), 1)
    precision = len(matched) / max(len(retrieved), 1)
    return RagQualityCaseResult(
        case_id=case.case_id,
        retrieved_chunk_ids=retrieved_ids,
        expected_chunk_ids=case.expected_chunk_ids,
        recall_at_k=round(recall, 3),
        precision_at_k=round(precision, 3),
        fallback=fallback,
        passed=recall > 0 and not fallback,
    )


def run_rag_quality_check(
    cases: list[RagQualityCase] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> RagQualitySummary:
    """Run the fixed set and return a lightweight retrieval-quality summary."""

    quality_cases = cases if cases is not None else load_quality_cases()
    results = [
        evaluate_quality_case(case, embedding_client=embedding_client)
        for case in quality_cases
    ]
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    average_recall = sum(result.recall_at_k for result in results) / max(total, 1)
    average_precision = sum(result.precision_at_k for result in results) / max(total, 1)
    return RagQualitySummary(
        total_cases=total,
        passed_cases=passed,
        average_recall_at_k=round(average_recall, 3),
        average_precision_at_k=round(average_precision, 3),
        results=results,
    )
