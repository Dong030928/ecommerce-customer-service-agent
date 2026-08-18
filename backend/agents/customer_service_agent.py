"""Agent orchestration for intent classification and grounded responses."""

from __future__ import annotations

import re

import httpx

from api.schemas import ChatRequest, ChatResponse, Intent, IntentResult
from models.answer_client import compose_grounded_answer
from models.classifier_client import classify_intent_with_model


def first_matched_keywords(message: str, keywords: list[str]) -> list[str]:
    """Return matched keywords so the classification remains observable."""

    return [keyword for keyword in keywords if keyword in message]


INTENT_RULES: list[tuple[Intent, list[str], str]] = [
    ("complaint", ["投诉", "举报", "赔偿", "曝光", "315", "别踢皮球"], "用户表达了投诉、赔偿或强烈不满，规则高置信标记为投诉类消息。"),
    ("refund_request", ["退款", "退货", "取消订单", "坏了", "无法开机", "质量问题"], "用户在询问退款、退货或质量问题，规则高置信标记为售后退款类消息。"),
    ("order_query", ["订单", "物流", "快递", "发货", "到哪", "运单"], "用户在询问订单或物流状态，规则高置信标记为订单查询类消息。"),
    ("promotion_consult", ["优惠", "活动", "会员价", "券", "满减", "折扣"], "用户在询问优惠或活动，规则高置信标记为活动咨询类消息。"),
    ("product_consult", ["耳机", "充电器", "音箱", "推荐", "哪个好"], "用户在询问商品或推荐，规则高置信标记为商品咨询类消息。"),
    ("general_chat", ["你好", "您好", "在吗", "谢谢"], "用户只是普通问候，规则高置信标记为普通聊天。"),
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
        prefix = message[max(0, start - 8):start]
        if not NEGATION_PREFIX_PATTERN.search(prefix):
            return False
        start = message.find(keyword, start + len(keyword))
    return found


def build_rule_evidence(message: str) -> list[tuple[Intent, list[str], list[str], str]]:
    """Collect positive and negated evidence before choosing an intent."""

    evidence = []
    for intent, keywords, explanation in INTENT_RULES:
        matched = first_matched_keywords(message, keywords)
        negated = [keyword for keyword in matched if is_negated_keyword(message, keyword)]
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

    refund = next((item for item in active_evidence if item[0] == "refund_request"), None)
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
        if any(keyword not in CONTEXT_KEYWORDS.get(item[0], set()) for keyword in item[1])
    ]
    if negated_keywords and not active_evidence:
        return IntentResult(
            intent="unknown",
            source="rules",
            confidence=0.65,
            matched_keywords=[],
            explanation="规则只发现被明确否定的意图，不能把它当成用户真实诉求，交给分类模型复核。",
        )
    if len(core_evidence) > 1 or (not core_evidence and len(active_evidence) > 1) or negated_keywords:
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
        "complaint": "我已经先把这条消息识别为投诉类问题。当前版本还没有接入人工流转和赔偿处理，不能直接承诺处理结果。",
        "refund_request": "我已经先把这条消息识别为退款或售后类问题。当前版本还没有接入售后规则和订单状态，不能直接判断是否可退。",
        "order_query": "我已经先把这条消息识别为订单或物流查询。当前版本还没有接入订单工具，不能编造物流节点。",
        "promotion_consult": "我已经先把这条消息识别为优惠活动咨询。当前版本还没有接入活动规则，不能承诺具体优惠。",
        "product_consult": "我已经先把这条消息识别为商品咨询。当前版本还没有接入产品知识库，不能编造商品卖点。",
        "general_chat": "你好，我是电商平台 AI 客服。现在我已经能把用户问题分到一个粗意图里。",
        "unknown": "我还不能确定这条消息属于哪类客服问题，只能先标记为 unknown。",
    }
    return answers[intent_result.intent]


class CustomerServiceAgent:
    """Customer-service agent with structured intent and answer boundaries."""

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
    ) -> None:
        self._message_count_by_session: dict[str, int] = {}
        self._classifier_http_client = classifier_http_client
        self._classifier_api_key = classifier_api_key
        self._classifier_base_url = classifier_base_url
        self._classifier_model_name = classifier_model_name
        self._answer_http_client = answer_http_client
        self._answer_api_key = answer_api_key
        self._answer_base_url = answer_base_url
        self._answer_model_name = answer_model_name

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Classify one message, generate an answer, and expose public state."""

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
        fallback_answer = build_answer(intent_result)
        model_answer = compose_grounded_answer(
            user_message=request.user_message,
            deterministic_answer=fallback_answer,
            intent_result=intent_result,
            http_client=self._answer_http_client,
            api_key=self._answer_api_key,
            base_url=self._answer_base_url,
            model=self._answer_model_name,
        )
        reasoning_summary = [
            "后端接收 ChatRequest，保持 user_message 与可信 runtime_* 分离。",
            "系统采用高置信规则优先、轻量分类模型兜底，生成稳定的粗粒度 intent。",
            "分类结果只用于消息分拣，不代表售后操作已经执行。",
            "IntentResult 经过 Pydantic 校验后才进入 ChatResponse。",
        ]
        session_state = {
            "agent_version": "0.3.0",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            "model_answer": model_answer.model_dump(),
            "next_gap": "系统知道消息大类，但还没有业务工具或工作流来决定并执行下一步动作。",
        }
        return ChatResponse(
            session_id=request.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            reasoning_summary=reasoning_summary,
            session_state=session_state,
        )
