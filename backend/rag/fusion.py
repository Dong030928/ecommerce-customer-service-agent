"""Weighted reciprocal-rank fusion for hybrid retrieval routes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


FUSION_VERSION = "chunk_rrf_v1"
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    score: float
    contributions: dict[str, dict[str, float | int]]


def validate_rrf_parameters(
    routes: Sequence[str],
    k: int,
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate configuration and resolve one non-negative weight per route."""

    if type(k) is not int or k <= 0:
        raise ValueError("rrf_k 必须是正整数。")
    if weights is not None and set(weights) - set(routes):
        raise ValueError("route_weights 包含未知检索路线。")
    resolved: dict[str, float] = {}
    for route in routes:
        value = (weights or {}).get(route, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("检索路线权重必须是有限的非负数。")
        resolved[route] = float(value)
    if resolved and not any(resolved.values()):
        raise ValueError("至少一条检索路线的权重必须大于 0。")
    return resolved


def fuse_rankings(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse route-local ranks without mixing incomparable raw scores."""

    resolved = validate_rrf_parameters(list(rankings), k, weights)
    evidence: dict[str, dict[str, dict[str, float | int]]] = {}
    for route in sorted(rankings):
        if resolved[route] == 0:
            continue
        unique_ids = list(dict.fromkeys(rankings[route]))
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in unique_ids):
            raise ValueError("chunk_id 必须是非空字符串。")
        for rank, chunk_id in enumerate(unique_ids, start=1):
            evidence.setdefault(chunk_id, {})[route] = {
                "rank": rank,
                "weight": resolved[route],
                "contribution": resolved[route] / (k + rank),
            }
    hits = [
        FusedHit(
            chunk_id=chunk_id,
            score=math.fsum(
                item["contribution"] for item in contributions.values()
            ),
            contributions=contributions,
        )
        for chunk_id, contributions in evidence.items()
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))
