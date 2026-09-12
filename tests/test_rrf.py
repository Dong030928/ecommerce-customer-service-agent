"""RRF arithmetic and integration tests without external services."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from api.schemas import KnowledgeChunk, KnowledgeHit  # noqa: E402
from rag.fusion import FUSION_VERSION, fuse_rankings  # noqa: E402
from rag.hybrid_retrieval import fuse_candidate_routes  # noqa: E402


def hit(
    chunk_id: str,
    *,
    vector_score: float | None = None,
    keyword_score: float | None = None,
    source: str,
) -> KnowledgeHit:
    score = max(vector_score or 0.0, keyword_score or 0.0)
    return KnowledgeHit(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            document_title="测试知识",
            source_path="knowledge/test.md",
            section="测试",
            text=f"{chunk_id} 内容",
        ),
        score=score,
        vector_score=vector_score,
        keyword_score=keyword_score,
        retrieval_sources=[source],
    )


class FusionTests(unittest.TestCase):
    def test_hand_calculated_multi_route_order(self) -> None:
        hits = fuse_rankings(
            {"original_vector": ["A", "B"], "keyword": ["B", "C"]}
        )

        self.assertEqual([item.chunk_id for item in hits], ["B", "A", "C"])
        self.assertAlmostEqual(hits[0].score, 1 / 62 + 1 / 61)
        self.assertEqual(hits[0].contributions["original_vector"]["rank"], 2)

    def test_deduplication_and_route_order_are_deterministic(self) -> None:
        first = fuse_rankings(
            {"original_vector": ["B", "B", "A"], "keyword": ["A", "B"]}
        )
        second = fuse_rankings(
            {"keyword": ["A", "B"], "original_vector": ["B", "A"]}
        )

        self.assertEqual(first, second)
        self.assertEqual([item.chunk_id for item in first], ["A", "B"])

    def test_invalid_configuration_is_rejected(self) -> None:
        for k in (True, 0, -1, 1.5, math.inf):
            with self.subTest(k=k), self.assertRaises(ValueError):
                fuse_rankings({"vector": ["A"]}, k=k)  # type: ignore[arg-type]
        for weights in (
            {"vector": -1},
            {"vector": math.nan},
            {"vector": math.inf},
            {"vector": True},
            {"vector": 0},
            {"unknown": 1},
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fuse_rankings({"vector": ["A"]}, weights=weights)

    def test_three_routes_fuse_without_overwriting_raw_scores(self) -> None:
        candidates, debug = fuse_candidate_routes(
            [
                hit("A", vector_score=0.9, source="original_vector"),
                hit("B", vector_score=0.8, source="original_vector"),
            ],
            [hit("B", vector_score=0.7, source="rewritten_vector")],
            [
                hit("B", keyword_score=0.6, source="keyword"),
                hit("C", keyword_score=0.5, source="keyword"),
            ],
        )

        self.assertEqual([item.chunk.chunk_id for item in candidates], ["B", "A", "C"])
        self.assertEqual(debug["fusion_method"], "rrf")
        self.assertEqual(debug["fusion_version"], FUSION_VERSION)
        self.assertEqual(debug["source_scores"]["B"]["contributions"]["keyword"]["rank"], 1)
        self.assertEqual(candidates[0].vector_score, 0.8)
        self.assertEqual(candidates[0].keyword_score, 0.6)
        self.assertGreater(candidates[0].fusion_score or 0.0, candidates[1].fusion_score or 0.0)

    def test_zero_weight_route_does_not_contribute(self) -> None:
        candidates, debug = fuse_candidate_routes(
            [hit("A", vector_score=0.9, source="original_vector")],
            [],
            [hit("B", keyword_score=0.9, source="keyword")],
            route_weights={
                "original_vector": 0,
                "rewritten_vector": 0,
                "keyword": 1,
            },
        )

        self.assertEqual([item.chunk.chunk_id for item in candidates], ["B"])
        self.assertEqual(debug["fused_chunk_ids"], ["B"])


if __name__ == "__main__":
    unittest.main()
