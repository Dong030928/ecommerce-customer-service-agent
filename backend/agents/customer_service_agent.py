"""Agent orchestration for intent classification and grounded vector-RAG answers."""

from __future__ import annotations

import re

import httpx

from api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    Intent,
    IntentResult,
    KnowledgeHit,
)
from config.settings import (
    LOW_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
    TOP_K,
)
from cost.observer import build_cost_summary
from embeddings.client import EmbeddingClient, read_embedding_model_name
from models.classifier_client import classify_intent_with_model
from models.llm_client import ModelAnswerResult, call_chat_model
from rag.knowledge_base import load_knowledge_chunks
from rag.quality import is_low_confidence, run_rag_quality_check
from rag.retrieval import retrieve_knowledge


def first_matched_keywords(message: str, keywords: list[str]) -> list[str]:
    """Return matched keywords so the classification remains observable."""

    return [keyword for keyword in keywords if keyword in message]


INTENT_RULES: list[tuple[Intent, list[str], str]] = [
    (
        "complaint",
        ["投诉", "举报", "赔偿", "曝光", "315", "别踢皮球"],
        "用户表达了投诉、赔偿或强烈不满，规则高置信标记为投诉类消息。",
    ),
    (
        "refund_request",
        ["退款", "退货", "取消订单", "坏了", "无法开机", "质量问题"],
        "用户在询问退款、退货或质量问题，规则高置信标记为售后退款类消息。",
    ),
    (
        "order_query",
        ["订单", "物流", "快递", "发货", "到哪", "运单"],
        "用户在询问订单或物流状态，规则高置信标记为订单查询类消息。",
    ),
    (
        "promotion_consult",
        ["优惠", "活动", "会员价", "券", "满减", "折扣"],
        "用户在询问优惠或活动，规则高置信标记为活动咨询类消息。",
    ),
    (
        "product_consult",
        ["耳机", "充电器", "音箱", "推荐", "哪个好"],
        "用户在询问商品或推荐，规则高置信标记为商品咨询类消息。",
    ),
    (
        "general_chat",
        ["你好", "您好", "在吗", "谢谢"],
        "用户只是普通问候，规则高置信标记为普通聊天。",
    ),
]

CONTEXT_KEYWORDS: dict[Intent, set[str]] = {
    "order_query": {"订单"},
    "product_consult": {"耳机", "充电器", "音箱"},
}

NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:不是|并非|不想|不打算|不需要|无需|不用|不要)(?:想要|想|要|申请|办理)?$"
)
REFUND_GOAL_PATTERN = re.compile(
    r"(?:直接|马上|立刻|赶紧|帮我|给我|我要|我想|申请|办理|能否|能不能|可以|是否|还能)"
    r"[^，。！？]{0,8}(?:退款|退货|取消订单|退钱)|(?:退款|退货|取消订单|退钱)(?:吗|么|吧|！|。|$)"
)


def is_negated_keyword(message: str, keyword: str) -> bool:
    """Detect negated signals such as '不是要退款'."""

    start = message.find(keyword)
    found = False
    while start >= 0:
        found = True
        prefix = message[max(0, start - 8) : start]
        if not NEGATION_PREFIX_PATTERN.search(prefix):
            return False
        start = message.find(keyword, start + len(keyword))
    return found


def build_rule_evidence(message: str) -> list[tuple[Intent, list[str], list[str], str]]:
    """Collect positive and negated evidence before choosing an intent."""

    evidence = []
    for intent, keywords, explanation in INTENT_RULES:
        matched = first_matched_keywords(message, keywords)
        negated = [
            keyword for keyword in matched if is_negated_keyword(message, keyword)
        ]
        active = [keyword for keyword in matched if keyword not in negated]
        if matched:
            evidence.append((intent, active, negated, explanation))
    return evidence


def plan_intent_by_rules(user_message: str) -> IntentResult | None:
    """Handle high-confidence customer-service intents deterministically."""

    message = user_message.strip().lower()
    evidence = build_rule_evidence(message)
    active_evidence = [item for item in evidence if item[1]]
    negated_keywords = [keyword for _, _, negated, _ in evidence for keyword in negated]

    complaint = next((item for item in active_evidence if item[0] == "complaint"), None)
    if complaint:
        return IntentResult(
            intent="complaint",
            source="rules",
            confidence=0.95,
            matched_keywords=complaint[1],
            explanation=complaint[3],
        )

    refund = next(
        (item for item in active_evidence if item[0] == "refund_request"), None
    )
    if refund and REFUND_GOAL_PATTERN.search(message):
        return IntentResult(
            intent="refund_request",
            source="rules",
            confidence=0.95,
            matched_keywords=refund[1],
            explanation=refund[3],
        )

    core_evidence = [
        item
        for item in active_evidence
        if any(
            keyword not in CONTEXT_KEYWORDS.get(item[0], set()) for keyword in item[1]
        )
    ]
    if negated_keywords and not active_evidence:
        return IntentResult(
            intent="unknown",
            source="rules",
            confidence=0.65,
            matched_keywords=[],
            explanation="规则只发现被明确否定的意图，不能把它当成用户真实诉求，交给分类模型复核。",
        )
    if (
        len(core_evidence) > 1
        or (not core_evidence and len(active_evidence) > 1)
        or negated_keywords
    ):
        primary = core_evidence[0] if core_evidence else active_evidence[0]
        intents = "、".join(item[0] for item in core_evidence or active_evidence)
        return IntentResult(
            intent=primary[0],
            source="rules",
            confidence=0.65,
            matched_keywords=primary[1],
            explanation=f"规则发现多个意图或否定表达（{intents}），证据不足以高置信直出，交给分类模型复核。",
        )

    if len(active_evidence) == 1:
        only_intent, active, _, _ = active_evidence[0]
        if not core_evidence:
            return IntentResult(
                intent=only_intent,
                source="rules",
                confidence=0.72,
                matched_keywords=active,
                explanation="当前只命中商品或订单等上下文实体，尚不能高置信判断用户真正诉求。",
            )

    for intent, keywords, explanation in INTENT_RULES:
        matched = first_matched_keywords(message, keywords)
        if matched:
            return IntentResult(
                intent=intent,
                source="rules",
                confidence=0.95,
                matched_keywords=matched,
                explanation=explanation,
            )
    return None


def classify_intent(
    user_message: str,
    *,
    http_client: httpx.Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> IntentResult:
    """Classify a message with high-confidence rules and a model fallback."""

    rule_result = plan_intent_by_rules(user_message)
    if rule_result and rule_result.confidence >= 0.85:
        return rule_result
    model_result = classify_intent_with_model(
        user_message,
        http_client=http_client,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    if model_result:
        return model_result
    return IntentResult(
        intent="unknown",
        source="rules_fallback",
        confidence=0.3,
        matched_keywords=[],
        explanation="规则没有高置信命中，分类模型也不可用或输出无效，先标记为 unknown。",
    )


def build_answer(intent_result: IntentResult) -> str:
    """Build a safe fallback answer for each coarse intent."""

    answers = {
        "complaint": "我已经识别到你的投诉诉求，但当前无法调用知识检索，暂时不能承诺赔偿或处理结果。",
        "refund_request": "我已经识别到你的退款或售后诉求，但当前没有检索到可核验规则，暂时不能判断是否可退。",
        "order_query": "我已经识别到订单或物流查询，但实时状态需要订单工具，不能根据知识库编造物流节点。",
        "promotion_consult": "我已经识别到优惠活动咨询，但当前没有检索到可核验规则，暂时不能承诺具体优惠。",
        "product_consult": "我已经识别到商品咨询，但当前没有检索到可核验资料，暂时不能编造商品卖点。",
        "general_chat": "你好，我是电商平台 AI 客服，可以协助解答商品、活动、订单和售后相关问题。",
        "unknown": "我还不能确定这条消息属于哪类客服问题，请补充商品、订单或售后诉求。",
    }
    return answers[intent_result.intent]


def build_rag_fallback(intent_result: IntentResult, retrieved_count: int) -> str:
    """Fail closed when retrieval or grounded generation is unavailable."""

    if retrieved_count:
        return f"我已找到 {retrieved_count} 条相关知识，但当前无法调用回答模型，不能据此直接承诺业务结果。"
    return build_answer(intent_result)


def build_low_confidence_answer() -> str:
    """Ask for clarification instead of turning a weak hit into a rule."""

    return (
        "我现在没有找到足够可靠的平台规则依据，不能直接给出结论。"
        "请补充订单状态、商品名称或活动页面信息；如果问题涉及售后争议，建议转人工继续核验。"
    )


def build_citations(hits: list[KnowledgeHit]) -> list[Citation]:
    """Convert only actual retrieval hits into public citations."""

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
    hits: list[KnowledgeHit],
) -> list[dict[str, str]]:
    """Render a minimal grounded prompt without trusted runtime identity values."""

    knowledge = "\n\n".join(
        f"[{index}] {hit.chunk.document_title} / {hit.chunk.section}\n"
        f"chunk_id={hit.chunk.chunk_id}\n{hit.chunk.text}"
        for index, hit in enumerate(hits, start=1)
    )
    system_content = (
        "你是电商平台 AI 客服。只能依据下方检索知识回答，不得编造活动、价格、库存、"
        "订单、物流或售后结论；知识不足时明确说明。回答简洁，并使用 [C1]、[C2] 标注引用。"
        f"\n当前意图：{intent_result.intent}\n检索知识：\n{knowledge}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]


class CustomerServiceAgent:
    """Customer-service agent with vector retrieval and verifiable citations."""

    def __init__(
        self,
        *,
        classifier_http_client: httpx.Client | None = None,
        classifier_api_key: str | None = None,
        classifier_base_url: str | None = None,
        classifier_model_name: str | None = None,
        answer_http_client: httpx.Client | None = None,
        answer_api_key: str | None = None,
        answer_base_url: str | None = None,
        answer_model_name: str | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._message_count_by_session: dict[str, int] = {}
        self._cost_events_by_session: dict[str, list[dict]] = {}
        self._classifier_http_client = classifier_http_client
        self._classifier_api_key = classifier_api_key
        self._classifier_base_url = classifier_base_url
        self._classifier_model_name = classifier_model_name
        self._answer_http_client = answer_http_client
        self._answer_api_key = answer_api_key
        self._answer_base_url = answer_base_url
        self._answer_model_name = answer_model_name
        self._embedding_client = embedding_client

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Classify, retrieve vectors, generate a grounded answer, and cite hits."""

        self._message_count_by_session[request.session_id] = (
            self._message_count_by_session.get(request.session_id, 0) + 1
        )
        message_count = self._message_count_by_session[request.session_id]
        intent_result = classify_intent(
            request.user_message,
            http_client=self._classifier_http_client,
            api_key=self._classifier_api_key,
            base_url=self._classifier_base_url,
            model=self._classifier_model_name,
        )

        raw_hits: list[KnowledgeHit] = []
        reliable_hits: list[KnowledgeHit] = []
        retrieval_error: str | None = None
        answer_path = "general_chat"
        low_confidence = False
        if intent_result.intent != "general_chat":
            try:
                raw_hits = retrieve_knowledge(
                    request.user_message, embedding_client=self._embedding_client
                )
                low_confidence = is_low_confidence(raw_hits)
                reliable_hits = [] if low_confidence else raw_hits
                answer_path = (
                    "grounded_model" if reliable_hits else "low_confidence_fallback"
                )
            except (
                RuntimeError,
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                retrieval_error = exc.__class__.__name__
                low_confidence = True
                answer_path = "retrieval_unavailable"

        citations = build_citations(reliable_hits)
        messages = render_rag_messages(
            request.user_message,
            intent_result,
            reliable_hits,
        )
        if intent_result.intent == "general_chat":
            fallback_answer = build_answer(intent_result)
        elif reliable_hits:
            fallback_answer = build_rag_fallback(intent_result, len(reliable_hits))
        else:
            fallback_answer = build_low_confidence_answer()

        if reliable_hits:
            model_answer = call_chat_model(
                messages,
                fallback_answer=fallback_answer,
                http_client=self._answer_http_client,
                api_key=self._answer_api_key,
                base_url=self._answer_base_url,
                model=self._answer_model_name,
            )
            if not model_answer.used_model:
                answer_path = "answer_model_fallback"
        else:
            model_answer = ModelAnswerResult(
                answer=fallback_answer, fallback_reason=answer_path
            )

        cost_summary = build_cost_summary(
            messages, model_answer.answer, model_answer.usage
        )
        event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
            "cost_summary": cost_summary.model_dump(),
        }
        self._cost_events_by_session.setdefault(request.session_id, []).append(event)
        chunks = load_knowledge_chunks()
        top_score = raw_hits[0].score if raw_hits else 0.0
        if intent_result.intent == "general_chat":
            confidence_level = "not_applicable"
            low_confidence_action = "general_chat_without_citations"
        elif retrieval_error:
            confidence_level = "unavailable"
            low_confidence_action = "clarify_or_handoff"
        elif low_confidence:
            confidence_level = "low"
            low_confidence_action = "clarify_or_handoff"
        else:
            confidence_level = "high"
            low_confidence_action = "answer_with_citations"

        quality_summary = None
        quality_error: str | None = retrieval_error
        quality_status = (
            "skipped_general_chat"
            if intent_result.intent == "general_chat"
            else "skipped_retrieval_unavailable"
        )
        if intent_result.intent != "general_chat" and retrieval_error is None:
            try:
                quality_summary = run_rag_quality_check(
                    embedding_client=self._embedding_client
                )
                quality_status = "completed"
            except (
                RuntimeError,
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                quality_error = exc.__class__.__name__
                quality_status = "unavailable"

        reasoning_summary = [
            "后端保持 user_message 与可信 runtime_* 分离，外部模型只接收业务问题。",
            "系统沿用规则优先、轻量分类模型兜底，得到稳定的结构化 intent。",
            f"向量检索从 {len(chunks)} 个知识块召回 {len(raw_hits)} 个候选，再用 {LOW_CONFIDENCE_THRESHOLD} 判断是否足以回答。",
            f"质量门槛保留 {len(reliable_hits)} 个可靠命中，并生成 {len(citations)} 条 citation。",
            f"本轮 token 来源为 {cost_summary.token_source}，总 token 为 {cost_summary.total_tokens}。",
        ]
        session_state = {
            "agent_version": "0.8.0",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            "model_answer": model_answer.model_dump(),
            "rag": {
                "mode": "quality_checked_vector_retrieval",
                "retrieval_strategy": "embedding_cosine_similarity",
                "vector_search": True,
                "embedding_model": read_embedding_model_name(),
                "top_k": TOP_K,
                "score_threshold": RETRIEVAL_SCORE_THRESHOLD,
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "document_count": len({chunk.source_path for chunk in chunks}),
                "chunk_count": len(chunks),
                "raw_retrieved_count": len(raw_hits),
                "retrieved_count": len(reliable_hits),
                "citation_count": len(citations),
                "confidence_level": confidence_level,
                "top_score": top_score,
                "low_confidence_action": low_confidence_action,
                "answer_path": answer_path,
                "raw_chunk_ids": [hit.chunk.chunk_id for hit in raw_hits],
                "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
                "scores": {hit.chunk.chunk_id: hit.score for hit in raw_hits},
                "retrieval_error": retrieval_error,
            },
            "rag_quality": {
                "status": quality_status,
                "case_count": quality_summary.total_cases if quality_summary else 0,
                "passed_cases": quality_summary.passed_cases if quality_summary else 0,
                "average_recall_at_k": (
                    quality_summary.average_recall_at_k if quality_summary else 0.0
                ),
                "average_precision_at_k": (
                    quality_summary.average_precision_at_k if quality_summary else 0.0
                ),
                "error": quality_error,
            },
            "cost_log": {
                "event_count": len(self._cost_events_by_session[request.session_id]),
                "latest": event,
            },
            "next_gap": "当前只有固定问题集和阈值兜底；下一步需要继续提升召回、排序质量，并建立完整 Evaluation 回归平台。",
        }
        return ChatResponse(
            session_id=request.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            citations=citations,
            cost_summary=cost_summary,
            reasoning_summary=reasoning_summary,
            session_state=session_state,
        )
