"""In-process vector index and cosine-similarity retrieval."""

from __future__ import annotations

import math
import weakref

from api.schemas import KnowledgeChunk, KnowledgeHit, VectorRecord
from config.settings import CANDIDATE_K, RETRIEVAL_SCORE_THRESHOLD
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
    embedding_texts = [chunk_embedding_text(chunk) for chunk in source_chunks]
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


def chunk_embedding_text(chunk: KnowledgeChunk) -> str:
    """Build the source-aware text sent to the embedding model."""

    return " ".join(
        [chunk.document_title, chunk.section, " ".join(chunk.keywords), chunk.text]
    )


def retrieve_candidates(
    query: str,
    *,
    top_k: int = CANDIDATE_K,
    threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    embedding_client: EmbeddingClient | None = None,
) -> list[KnowledgeHit]:
    """Retrieve the wider vector candidate set used by the reranker."""

    query_embedding = embed_text(query, embedding_client)
    asks_for_history = query_asks_for_history(query)
    hits: list[KnowledgeHit] = []
    for record in get_vector_store(embedding_client):
        if not should_include_chunk_for_query(record.chunk, asks_for_history):
            continue
        score = cosine_similarity(query_embedding, record.embedding)
        if score >= threshold:
            rounded_score = round(max(0.0, min(1.0, score)), 3)
            hits.append(
                KnowledgeHit(
                    chunk=record.chunk,
                    score=rounded_score,
                    vector_score=rounded_score,
                )
            )
    return sorted(
        hits,
        key=lambda hit: hit.vector_score if hit.vector_score is not None else hit.score,
        reverse=True,
    )[:top_k]


def merge_candidates(*candidate_groups: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """Merge original-query and rewritten-query candidates by best vector score."""

    merged: dict[str, KnowledgeHit] = {}
    for group in candidate_groups:
        for hit in group:
            current = merged.get(hit.chunk.chunk_id)
            hit_score = hit.vector_score if hit.vector_score is not None else hit.score
            current_score = (
                current.vector_score
                if current is not None and current.vector_score is not None
                else current.score if current is not None else -1.0
            )
            if current is None or hit_score > current_score:
                merged[hit.chunk.chunk_id] = hit
    return sorted(
        merged.values(),
        key=lambda hit: hit.vector_score if hit.vector_score is not None else hit.score,
        reverse=True,
    )[:CANDIDATE_K]
