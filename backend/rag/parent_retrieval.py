"""Expand reliable children to bounded parent evidence in the pinned snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

from api.schemas import KnowledgeHit, KnowledgeIndex
from config.settings import FINAL_TOP_K, LOW_CONFIDENCE_THRESHOLD
from cost.observer import estimate_tokens
from rag.knowledge_base import query_asks_for_history, should_include_chunk_for_query
from safety.prompt_guard import sanitize_text


@dataclass
class ParentExpansionOutcome:
    hits: list[KnowledgeHit]
    trace: dict[str, Any]


def evidence_text(hit: KnowledgeHit) -> str:
    """The exact sanitized body used by both the prompt and public citation."""
    return sanitize_text(hit.parent.text if hit.parent is not None else hit.chunk.text)[0]


def expand_parent_evidence(
    hits: list[KnowledgeHit],
    index: KnowledgeIndex,
    query: str,
    allowed_domains: list[str],
    *,
    preferred_domains: tuple[str, ...] = (),
    active_only: bool = False,
    top_k: int | None = None,
    char_budget: int | None = None,
) -> ParentExpansionOutcome:
    started = perf_counter()
    top_k = int(os.getenv("AGENT_RAG_PARENT_TOP_K", FINAL_TOP_K)) if top_k is None else top_k
    char_budget = int(os.getenv("AGENT_RAG_EVIDENCE_CHAR_BUDGET", 3200)) if char_budget is None else char_budget
    if type(top_k) is not int or type(char_budget) is not int or top_k <= 0 or char_budget <= 0:
        raise ValueError("父块数量和证据字符预算必须为正整数。")
    grouped: dict[str, KnowledgeHit] = {}
    skipped: list[dict[str, str]] = []
    asks_history = query_asks_for_history(query)
    # Stable ties preserve the upstream RRF/reranker order.
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        chunk = hit.chunk
        reason = None
        if hit.score < LOW_CONFIDENCE_THRESHOLD:
            reason = "low_confidence"
        elif hit.index_version is not None and hit.index_version != index.version:
            reason = "stale_index_version"
        elif index.chunks_by_id.get(chunk.chunk_id) != chunk:
            reason = "stale_or_unknown_child"
        elif chunk.metadata.get("domain") not in allowed_domains:
            reason = "domain_filtered"
        elif (active_only and chunk.effective_status != "active") or not should_include_chunk_for_query(chunk, asks_history):
            reason = "status_filtered"
        parent = index.parents_by_id.get(chunk.parent_id)
        if reason is None and parent is None:
            reason = "missing_parent"
        if reason is None and (
            parent.source_path != chunk.source_path
            or parent.document_title != chunk.document_title
            or parent.section != chunk.section
            or parent.keywords != chunk.keywords
            or parent.effective_status != chunk.effective_status
            or parent.metadata.get("domain") != chunk.metadata.get("domain")
            or chunk.parent_end is None
            or not 0 <= chunk.parent_start < chunk.parent_end <= len(parent.text)
            or parent.text[chunk.parent_start:chunk.parent_end] != chunk.text
        ):
            reason = "parent_child_mismatch"
        if reason:
            skipped.append({"chunk_id": chunk.chunk_id, "reason": reason})
            continue
        existing = grouped.get(parent.parent_id)
        if existing is None:
            grouped[parent.parent_id] = hit.model_copy(deep=True, update={
                "parent": parent.model_copy(deep=True), "matched_child_ids": [chunk.chunk_id],
                "index_version": index.version,
            })
        elif chunk.chunk_id not in existing.matched_child_ids:
            existing.matched_child_ids.append(chunk.chunk_id)
    # Each parent inherits only its best child's score, never a sum biased by size.
    ordered = list(grouped.values())
    preferred: list[KnowledgeHit] = []
    for domain in preferred_domains:
        match = next((hit for hit in ordered if hit.parent.metadata.get("domain") == domain), None)
        if match is not None and match not in preferred:
            preferred.append(match)
    ordered = preferred + [hit for hit in ordered if hit not in preferred]
    selected: list[KnowledgeHit] = []
    body_chars = 0
    for hit in ordered:
        body = evidence_text(hit)
        if not body.strip():
            reason = "empty_sanitized_evidence"
        elif len(selected) >= top_k:
            reason = "parent_top_k"
        elif body_chars + len(body) > char_budget:
            reason = "evidence_char_budget"
        else:
            reason = None
        if reason:
            skipped.append({"parent_id": hit.parent.parent_id, "reason": reason})
            continue
        selected.append(hit)
        body_chars += len(body)
    return ParentExpansionOutcome(hits=selected, trace={
        "index_version": index.version, "parent_top_k": top_k, "evidence_char_budget": char_budget,
        "input_child_count": len(hits), "reliable_parent_count": len(grouped),
        "selected_parent_ids": [hit.parent.parent_id for hit in selected],
        "matched_child_ids": [child for hit in selected for child in hit.matched_child_ids],
        "evidence_body_chars": body_chars,
        "estimated_evidence_body_tokens": sum(estimate_tokens(evidence_text(hit)) for hit in selected),
        "token_source": "local_estimate", "skipped": skipped,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
    })
