"""In-process vector index and cosine-similarity retrieval."""

from __future__ import annotations

import math
import weakref

from api.schemas import KnowledgeChunk, KnowledgeHit, VectorRecord
from config.settings import SCORE_THRESHOLD, TOP_K
from embeddings.client import (
    DEFAULT_EMBEDDING_CLIENT,
    EmbeddingClient,
    embed_text,
    embed_texts,
)
from rag.knowledge_base import (
    load_knowledge_chunks,
    query_asks_for_history,
    should_include_chunk_for_query,
)


_VECTOR_STORE_CACHE: weakref.WeakKeyDictionary[
    EmbeddingClient,
    list[VectorRecord],
] = weakref.WeakKeyDictionary()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity and reject mismatched vector dimensions."""

    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：left={len(left)}, right={len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def build_vector_store(
    chunks: list[KnowledgeChunk] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[VectorRecord]:
    """Batch-embed source chunks into an in-process vector index."""

    source_chunks = chunks if chunks is not None else load_knowledge_chunks()
    embedding_texts = [
        " ".join(
            [chunk.document_title, chunk.section, " ".join(chunk.keywords), chunk.text]
        )
        for chunk in source_chunks
    ]
    embeddings = embed_texts(embedding_texts, embedding_client)
    if len(embeddings) != len(source_chunks):
        raise ValueError("知识片段与 Embedding 数量不一致。")
    return [
        VectorRecord(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(source_chunks, embeddings)
    ]


def get_vector_store(
    embedding_client: EmbeddingClient | None = None,
) -> list[VectorRecord]:
    """Build the vector index lazily and cache it by client instance."""

    client = embedding_client or DEFAULT_EMBEDDING_CLIENT
    if client not in _VECTOR_STORE_CACHE:
        _VECTOR_STORE_CACHE[client] = build_vector_store(embedding_client=client)
    return _VECTOR_STORE_CACHE[client]


def retrieve_by_vector(
    query: str,
    *,
    top_k: int = TOP_K,
    threshold: float = SCORE_THRESHOLD,
    embedding_client: EmbeddingClient | None = None,
) -> list[KnowledgeHit]:
    """Return thresholded Top-K chunks for a query embedding."""

    query_embedding = embed_text(query, embedding_client)
    asks_for_history = query_asks_for_history(query)
    hits: list[KnowledgeHit] = []
    for record in get_vector_store(embedding_client):
        if not should_include_chunk_for_query(record.chunk, asks_for_history):
            continue
        score = cosine_similarity(query_embedding, record.embedding)
        if score >= threshold:
            hits.append(KnowledgeHit(chunk=record.chunk, score=round(score, 3)))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
