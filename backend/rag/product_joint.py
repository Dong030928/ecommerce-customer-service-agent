"""Balanced evidence selection for product Tool + RAG answers."""

from __future__ import annotations

from api.schemas import KnowledgeHit
from config.settings import FINAL_TOP_K, LOW_CONFIDENCE_THRESHOLD


def select_product_joint_hits(hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """Keep both product knowledge and promotion policy when both are reliable."""

    reliable = [
        hit
        for hit in hits
        if hit.score >= LOW_CONFIDENCE_THRESHOLD
        and hit.chunk.effective_status == "active"
    ]
    selected: list[KnowledgeHit] = []
    for domain in ("product", "promotion"):
        match = next(
            (
                hit
                for hit in reliable
                if str(hit.chunk.metadata.get("domain") or "") == domain
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    for hit in reliable:
        if len(selected) >= FINAL_TOP_K:
            break
        if hit not in selected:
            selected.append(hit)
    return selected[:FINAL_TOP_K]
