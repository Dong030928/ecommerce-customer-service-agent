"""Citation and prompt rendering after query rewrite and reranking."""

from __future__ import annotations

from api.schemas import Citation, IntentResult, KnowledgeHit, QueryRewrite


def build_citations(hits: list[KnowledgeHit]) -> list[Citation]:
    """Convert only final reliable hits into public citations."""

    return [
        Citation(
            citation_id=f"C{index}",
            source_title=hit.chunk.document_title,
            source_path=hit.chunk.source_path,
            section=hit.chunk.section,
            chunk_id=hit.chunk.chunk_id,
            score=hit.score,
            snippet=hit.chunk.text,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def render_rag_messages(
    user_message: str,
    intent_result: IntentResult,
    rewrite: QueryRewrite,
    hits: list[KnowledgeHit],
) -> list[dict[str, str]]:
    """Render final evidence without including trusted runtime identity values."""

    evidence = "\n\n".join(
        f"[{index}] {hit.chunk.document_title} / {hit.chunk.section}\n"
        f"chunk_id={hit.chunk.chunk_id} score={hit.score} "
        f"reasons={','.join(hit.rerank_reasons)}\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    )
    system_content = (
        "你是电商平台 AI 客服。只能依据重排后给出的知识回答，不得编造活动、价格、库存、"
        "订单、物流或售后结论；实时事实需要业务系统核验。回答简洁，并使用 [C1]、[C2] 标注引用。"
    )
    user_content = (
        f"用户原话：{user_message}\n"
        f"粗意图：{intent_result.intent}\n"
        f"检索改写：{rewrite.rewritten_query}\n"
        f"重排后的可靠知识：\n{evidence}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
