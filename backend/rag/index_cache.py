"""Versioned knowledge index and bounded retrieval/vector caches."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from threading import RLock

from api.schemas import (
    KnowledgeChunk,
    KnowledgeIndex,
    RetrievalCacheEntry,
    RetrievalPlan,
    VectorRecord,
)
from config.settings import RAG_RETRIEVAL_CACHE_MAX_ENTRIES
from rag.knowledge_base import load_knowledge_chunks
from rag.query_rewrite import normalize_query


_CACHE_LOCK = RLock()
_KNOWLEDGE_INDEX: KnowledgeIndex | None = None
_VECTOR_STORE_CACHE: dict[tuple[str, str], list[VectorRecord]] = {}
_RETRIEVAL_CACHE: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()


def build_knowledge_index(chunks: list[KnowledgeChunk]) -> KnowledgeIndex:
    """Build a deterministic index version and exact-keyword inverted index."""

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise ValueError("知识索引包含重复 chunk_id，拒绝构建不完整索引。")
    serialized = json.dumps(
        [chunk.model_dump(mode="json") for chunk in chunks],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    inverted_index: dict[str, list[str]] = {}
    for chunk in chunks:
        for keyword in chunk.keywords:
            inverted_index.setdefault(normalize_query(keyword), []).append(
                chunk.chunk_id
            )
    return KnowledgeIndex(
        version=f"idx-{fingerprint[:12]}",
        fingerprint=fingerprint,
        chunk_count=len(chunks),
        document_count=len({chunk.source_path for chunk in chunks}),
        chunks_by_id=chunks_by_id,
        inverted_index=inverted_index,
    )


def get_knowledge_index() -> KnowledgeIndex:
    """Lazily load the current in-process knowledge-index snapshot."""

    global _KNOWLEDGE_INDEX
    with _CACHE_LOCK:
        if _KNOWLEDGE_INDEX is None:
            _KNOWLEDGE_INDEX = build_knowledge_index(load_knowledge_chunks())
        return _KNOWLEDGE_INDEX


def rebuild_knowledge_index(
    chunks: list[KnowledgeChunk] | None = None,
) -> KnowledgeIndex:
    """Replace the index snapshot and invalidate all dependent caches."""

    global _KNOWLEDGE_INDEX
    source_chunks = chunks if chunks is not None else load_knowledge_chunks()
    rebuilt = build_knowledge_index(source_chunks)
    with _CACHE_LOCK:
        _KNOWLEDGE_INDEX = rebuilt
        _VECTOR_STORE_CACHE.clear()
        _RETRIEVAL_CACHE.clear()
    return rebuilt


def reset_index_and_cache() -> None:
    """Clear process-local index state for tests and controlled maintenance."""

    global _KNOWLEDGE_INDEX
    with _CACHE_LOCK:
        _KNOWLEDGE_INDEX = None
        _VECTOR_STORE_CACHE.clear()
        _RETRIEVAL_CACHE.clear()


def get_cached_vector_store(
    index_version: str,
    embedding_identity: str,
) -> list[VectorRecord] | None:
    with _CACHE_LOCK:
        return _VECTOR_STORE_CACHE.get((index_version, embedding_identity))


def store_vector_store(
    index_version: str,
    embedding_identity: str,
    records: list[VectorRecord],
) -> None:
    with _CACHE_LOCK:
        _VECTOR_STORE_CACHE[(index_version, embedding_identity)] = records


def retrieval_cache_key(
    plan: RetrievalPlan,
    index: KnowledgeIndex,
    embedding_identity: str,
) -> str:
    """Hash all inputs that can change retrieval candidates."""

    payload = json.dumps(
        {
            "index_version": index.version,
            "embedding_identity": embedding_identity,
            "scene": plan.scene,
            "allowed_domains": plan.allowed_domains,
            "original_query": normalize_query(plan.original_query),
            "rewritten_query": normalize_query(plan.rewritten_query),
            "keyword_terms": plan.keyword_terms,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def get_retrieval_cache_entry(cache_key: str) -> RetrievalCacheEntry | None:
    with _CACHE_LOCK:
        entry = _RETRIEVAL_CACHE.get(cache_key)
        if entry is not None:
            _RETRIEVAL_CACHE.move_to_end(cache_key)
        return entry


def store_retrieval_cache_entry(
    cache_key: str,
    entry: RetrievalCacheEntry,
) -> None:
    """Store one entry and evict the least-recently-used item when bounded."""

    with _CACHE_LOCK:
        _RETRIEVAL_CACHE[cache_key] = entry
        _RETRIEVAL_CACHE.move_to_end(cache_key)
        while len(_RETRIEVAL_CACHE) > RAG_RETRIEVAL_CACHE_MAX_ENTRIES:
            _RETRIEVAL_CACHE.popitem(last=False)


def cache_entry_count() -> int:
    with _CACHE_LOCK:
        return len(_RETRIEVAL_CACHE)
