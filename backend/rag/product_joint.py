"""Balanced parent evidence for product Tool + RAG answers."""

from __future__ import annotations

from api.schemas import KnowledgeHit, KnowledgeIndex
from rag.parent_retrieval import ParentExpansionOutcome, expand_parent_evidence


def expand_product_joint_evidence(
    hits: list[KnowledgeHit], index: KnowledgeIndex, query: str, allowed_domains: list[str],
) -> ParentExpansionOutcome:
    """Balance domains after parent deduplication, never truncate children first."""

    return expand_parent_evidence(
        hits, index, query, allowed_domains,
        preferred_domains=("product", "promotion"), active_only=True,
    )
