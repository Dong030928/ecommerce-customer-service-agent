"""Reproducible synthetic evidence/scale check; no provider calls or quality claims.

Run: .venv/Scripts/python.exe tests/benchmark_parent_child.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from api.schemas import KnowledgeHit, KnowledgeSection
from cost.observer import estimate_tokens
from rag.index_cache import build_knowledge_index
from rag.knowledge_base import split_into_chunks
from rag.parent_child import ChunkingConfig, split_section
from rag.parent_retrieval import evidence_text, expand_parent_evidence


def make_section(text: str, identity: str) -> KnowledgeSection:
    return KnowledgeSection(source_path=f"{identity}.md", document_title="合成政策",
                            section_index=1, section="规则", chunk_id=identity, text=text,
                            metadata={"domain": "shipping"})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = ChunkingConfig()
    comparisons = []
    for identity, main_rule, exception in (
        ("shipping", "现货48小时发货", "预售按商品页预计发货时间"),
        ("return", "七天无理由退货", "特殊商品需人工核验"),
        ("promotion", "会员价可叠加优惠券", "最终优惠以结算页为准"),
    ):
        text = main_rule + "。" + "通用说明。" * 200 + "\n\n例外：" + exception + "。"
        parents, children = split_section(make_section(text, identity), config)
        index = build_knowledge_index(children, parents, config)
        # Deliberately fix the recalled children; this isolates context expansion,
        # not semantic retrieval accuracy. The original flat Top-2 misses the tail.
        hits = [KnowledgeHit(chunk=child, score=0.9) for child in children[:2]]
        result = expand_parent_evidence(hits, index, "物流", ["shipping"])
        flat = split_into_chunks(text)[:2]
        expanded = [evidence_text(hit) for hit in result.hits]
        terms = [main_rule, exception]
        comparisons.append({
            "case": identity, "synthetic_controlled_children": True,
            "flat_top2_required_term_coverage": sum(term in "\n".join(flat) for term in terms) / len(terms),
            "parent_required_term_coverage": sum(term in "\n".join(expanded) for term in terms) / len(terms),
            "flat_body_chars": sum(map(len, flat)), "parent_body_chars": sum(map(len, expanded)),
            "flat_estimated_body_tokens": sum(map(estimate_tokens, flat)),
            "parent_estimated_body_tokens": result.trace["estimated_evidence_body_tokens"],
        })
        assert comparisons[-1]["parent_required_term_coverage"] == 1.0

    parents, children = [], []
    started = perf_counter()
    for n in range(100):
        p, c = split_section(make_section(("配送规则。" * 2400), f"scale-{n}"), config)
        parents.extend(p)
        children.extend(c)
    split_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    index = build_knowledge_index(children, parents, config)
    build_ms = (perf_counter() - started) * 1000
    hits = [KnowledgeHit(chunk=children[i], score=0.9, index_version=index.version)
            for i in (0, 1, 2, 45, 46, 47)]
    durations = []
    for _ in range(200):
        started = perf_counter()
        result = expand_parent_evidence(hits, index, "物流", ["shipping"])
        durations.append((perf_counter() - started) * 1000)
        assert result.trace["evidence_body_chars"] <= 3200
        assert len(result.hits) <= 2
    print(json.dumps({"note": "纯离线合成实验，覆盖率是预置条款覆盖，不是线上准确率/召回率；Token 是正文估算。",
                      "comparisons": comparisons,
                      "scale": {"documents": 100, "source_body_chars": 1200000,
                                "parents": len(parents), "children": len(children),
                                "split_ms": round(split_ms, 2), "build_ms": round(build_ms, 2),
                                "expansion_iterations": 200, "expansion_median_ms": round(median(durations), 3),
                                "expansion_p95_ms": round(sorted(durations)[189], 3),
                                "includes_model_network_or_vector_scan": False}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
