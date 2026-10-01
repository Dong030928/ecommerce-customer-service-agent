"""Versioned knowledge index and bounded retrieval/vector caches."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from threading import RLock

from api.schemas import (
    KnowledgeChunk,
    KnowledgeIndex,
    KnowledgeParent,
    RetrievalCacheEntry,
    RetrievalPlan,
    VectorRecord,
)
from config.settings import RAG_RETRIEVAL_CACHE_MAX_ENTRIES
from rag.knowledge_base import build_knowledge_corpus
from rag.parent_child import ChunkingConfig, located_metadata, slice_spans
from rag.query_rewrite import normalize_query
from rag.fusion import FUSION_VERSION


_CACHE_LOCK = RLock()
_KNOWLEDGE_INDEX: KnowledgeIndex | None = None
_VECTOR_STORE_CACHE: dict[tuple[str, str], list[VectorRecord]] = {}
_RETRIEVAL_CACHE: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()


def build_knowledge_index(
    chunks: list[KnowledgeChunk],
    parents: list[KnowledgeParent],
    config: ChunkingConfig | None = None,
) -> KnowledgeIndex:
    """Build a deterministic index version and exact-keyword inverted index."""

    if not chunks or not parents:
        raise ValueError("知识索引父块与子块不能为空。")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise ValueError("知识索引包含重复 chunk_id，拒绝构建不完整索引。")
    config = config or ChunkingConfig.from_env()
    parents_by_id = {parent.parent_id: parent for parent in parents}
    if len(parents_by_id) != len(parents):
        raise ValueError("知识索引包含重复 parent_id。")
    for parent in parents:
        if not parent.text.strip() or len(parent.text) > config.parent_size:
            raise ValueError(f"父块正文为空或超过配置上限：{parent.parent_id}")
        previous_end = 0
        for span in parent.source_spans:
            if not previous_end <= span.start < span.end <= len(parent.text):
                raise ValueError(f"父块来源偏移无效：{parent.parent_id}")
            previous_end = span.end
    for chunk in chunks:
        parent = parents_by_id.get(chunk.parent_id)
        if parent is None:
            raise ValueError(f"子块缺少父块：{chunk.chunk_id}")
        if (chunk.parent_end is None or not 0 <= chunk.parent_start < chunk.parent_end <= len(parent.text)
                or chunk.text != parent.text[chunk.parent_start:chunk.parent_end]
                or not chunk.text.strip() or len(chunk.text) > config.child_size):
            raise ValueError(f"子块正文或父块偏移不一致：{chunk.chunk_id}")
        for field in ("source_path", "document_title", "section", "effective_status", "keywords"):
            if getattr(chunk, field) != getattr(parent, field):
                raise ValueError(f"父子块 {field} 元数据不一致：{chunk.chunk_id}")
        for key in ("domain", "owner", "document_id", "source_format"):
            if chunk.metadata.get(key) != parent.metadata.get(key):
                raise ValueError(f"父子块 {key} 元数据不一致：{chunk.chunk_id}")
        if chunk.source_spans != slice_spans(parent.source_spans, chunk.parent_start, chunk.parent_end):
            raise ValueError(f"父子块页码偏移不一致：{chunk.chunk_id}")
        for unit in (parent, chunk):
            located = located_metadata(unit.metadata, unit.source_spans)
            if any(unit.metadata.get(key) != located.get(key) for key in ("page_number", "page_start", "page_end")):
                raise ValueError(f"父子块页码元数据不一致：{chunk.chunk_id}")
    if set(parents_by_id) != {chunk.parent_id for chunk in chunks}:
        raise ValueError("索引包含没有子块的父块。")
    serialized = json.dumps(
        # Input order participates in equal-score retrieval ties and cache semantics.
        {"parents": [parent.model_dump(mode="json") for parent in parents],
         "children": [chunk.model_dump(mode="json") for chunk in chunks],
         "splitting_config": config.as_dict()},
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
        chunks_by_id={key: value.model_copy(deep=True) for key, value in chunks_by_id.items()},
        inverted_index=inverted_index,
        parents_by_id={key: value.model_copy(deep=True) for key, value in parents_by_id.items()},
        splitting_config=config.as_dict(),
    )


def get_knowledge_index() -> KnowledgeIndex:
    """Lazily load the current in-process knowledge-index snapshot."""

    global _KNOWLEDGE_INDEX
    with _CACHE_LOCK:
        if _KNOWLEDGE_INDEX is None:
            config = ChunkingConfig.from_env()
            parents, chunks = build_knowledge_corpus(config)
            _KNOWLEDGE_INDEX = build_knowledge_index(chunks, parents, config)
        return _KNOWLEDGE_INDEX


def rebuild_knowledge_index(
    chunks: list[KnowledgeChunk] | None = None,
    parents: list[KnowledgeParent] | None = None,
    config: ChunkingConfig | None = None,
) -> KnowledgeIndex:
    """Replace the index snapshot and invalidate all dependent caches."""

    global _KNOWLEDGE_INDEX
    config = config or ChunkingConfig.from_env()
    if chunks is None and parents is None:
        parents, chunks = build_knowledge_corpus(config)
    elif chunks is None or parents is None:
        raise ValueError("重建索引必须同时提供完整父块与子块。")
    rebuilt = build_knowledge_index(chunks, parents, config)
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
        if _KNOWLEDGE_INDEX is not None and index_version != _KNOWLEDGE_INDEX.version:
            # An in-flight old snapshot may finish embedding after a rebuild.
            # It can use its records, but must not repopulate invalidated caches.
            return
        _VECTOR_STORE_CACHE[(index_version, embedding_identity)] = records


def retrieval_cache_key(
    plan: RetrievalPlan,
    index: KnowledgeIndex,
    embedding_identity: str,
    fusion_config: dict[str, object] | None = None,
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
            "fusion_version": FUSION_VERSION,
            "fusion_config": fusion_config or {},
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
        if _KNOWLEDGE_INDEX is not None and entry.index_version != _KNOWLEDGE_INDEX.version:
            return
        _RETRIEVAL_CACHE[cache_key] = entry
        _RETRIEVAL_CACHE.move_to_end(cache_key)
        while len(_RETRIEVAL_CACHE) > RAG_RETRIEVAL_CACHE_MAX_ENTRIES:
            _RETRIEVAL_CACHE.popitem(last=False)


def cache_entry_count() -> int:
    with _CACHE_LOCK:
        return len(_RETRIEVAL_CACHE)
