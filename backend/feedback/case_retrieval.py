"""Vector retrieval for recommending regression cases during feedback triage."""

from __future__ import annotations

import math
from typing import Protocol

from api.schemas import CaseRecommendation, FeedbackRequest
from embeddings.client import DEFAULT_EMBEDDING_CLIENT
from evals.runner import EvalCase
from observability.trace import TraceEventNormalizer
from safety.prompt_guard import sanitize_text


class CaseEmbeddingClient(Protocol):
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class CaseRecommendationService:
    """Embed incident evidence and cases, then return the nearest candidates."""

    def __init__(self, embedding_client: CaseEmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client or DEFAULT_EMBEDDING_CLIENT

    def recommend(
        self,
        feedback: FeedbackRequest,
        cases: list[EvalCase],
    ) -> list[CaseRecommendation]:
        if not cases:
            return []
        query = self._feedback_query(feedback)
        documents = [self._case_document(case) for case in cases]
        vectors = self.embedding_client.embed_many([query, *documents])
        if len(vectors) != len(documents) + 1:
            raise ValueError("Case Embedding 返回的向量数量不一致。")
        query_vector = vectors[0]
        ranked = sorted(
            zip(cases, vectors[1:]),
            key=lambda item: self._cosine_similarity(query_vector, item[1]),
            reverse=True,
        )[: feedback.top_k]
        return [
            CaseRecommendation(
                case_id=case.case_id,
                similarity_score=round(self._cosine_similarity(query_vector, vector), 6),
                scenario_summary=self._safe_text(case.user_message),
                source=case.source,
            )
            for case, vector in ranked
        ]

    @staticmethod
    def _feedback_query(feedback: FeedbackRequest) -> str:
        parts = [
            f"原始用户问题：{feedback.user_message}" if feedback.user_message else "",
            f"错误回答：{feedback.observed_answer}",
            f"用户反馈：{feedback.user_comment}",
        ]
        return CaseRecommendationService._safe_text("\n".join(part for part in parts if part))

    @staticmethod
    def _case_document(case: EvalCase) -> str:
        parts = [
            f"用户场景：{case.user_message}",
            f"期望回答信号：{'、'.join(case.expected_signals)}",
            f"期望工具：{'、'.join(case.expected_tools)}",
            f"期望引用：{'、'.join(case.expected_citations)}",
            f"期望链路：{'、'.join(case.expected_trace_events)}",
            f"禁止内容：{'、'.join(case.forbidden_text)}",
            f"来源：{case.source}",
        ]
        return "\n".join(part for part in parts if not part.endswith("："))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            raise ValueError("Case Embedding 向量维度不一致。")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    @staticmethod
    def _safe_text(text: str) -> str:
        return TraceEventNormalizer.sanitize(sanitize_text(text)[0])
