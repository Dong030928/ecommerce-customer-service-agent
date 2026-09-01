"""Citation and prompt rendering after query rewrite and reranking."""

from __future__ import annotations

from api.schemas import (
    Citation,
    IntentResult,
    KnowledgeHit,
    QueryRewrite,
    ToolObservation,
)


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


def render_product_tool_rag_messages(
    user_message: str,
    observations: list[ToolObservation],
    hits: list[KnowledgeHit],
) -> list[dict[str, str]]:
    """Render a joint prompt while keeping current and stable facts separated."""

    tool_evidence = "\n".join(
        observation.model_dump_json(
            include={"tool_name", "status", "summary", "facts", "next_action"}
        )
        for observation in observations
    )
    rag_evidence = "\n\n".join(
        f"[C{index}] {hit.chunk.document_title} / {hit.chunk.section}\n"
        f"chunk_id={hit.chunk.chunk_id} score={hit.score}\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "你是电商平台 AI 客服。当前价格、库存和活动状态只能引用 TOOL_OBSERVATIONS；"
                "商品卖点与平台活动规则只能引用 RAG_EVIDENCE，并使用 [C1]、[C2] 标注。"
                "两类证据冲突时，以工具的当前事实为准；不得把平台规则说成当前 SKU 必然享有的优惠，"
                "最终优惠以结算页为准。不得编造证据中不存在的商品、价格、库存或活动。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户原话：{user_message}\n"
                f"TOOL_OBSERVATIONS：\n{tool_evidence}\n\n"
                f"RAG_EVIDENCE：\n{rag_evidence}"
            ),
        },
    ]


def build_product_tool_rag_fallback(
    observations: list[ToolObservation],
    hits: list[KnowledgeHit],
) -> str:
    """Compose a deterministic joint answer from already-sanitized evidence."""

    tool_lines = [
        observation.summary
        for observation in observations
        if observation.status == "success"
    ]
    answer = "实时商品信息：" + " ".join(tool_lines)
    if hits:
        knowledge_lines = [
            f"[C{index}] {hit.chunk.text}"
            for index, hit in enumerate(hits, start=1)
        ]
        answer += " 知识库依据：" + " ".join(knowledge_lines)
    else:
        answer += " 暂未找到足够可靠的商品知识或活动规则，不能补充相关承诺。"
    return answer
