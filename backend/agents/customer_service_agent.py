"""Agent orchestration for intent classification and grounded vector-RAG answers."""

from __future__ import annotations

import re

import httpx

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ClarificationRequest,
    DegradationState,
    Intent,
    IntentResult,
    KnowledgeHit,
)
from config.settings import (
    CANDIDATE_K,
    FINAL_TOP_K,
    HYBRID_CANDIDATE_K,
    LOW_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
)
from cost.observer import build_cost_summary
from degradation.fallbacks import (
    degradation_from_tool_records,
    fallback_message,
    high_risk_degradation,
    is_high_risk_write_request,
)
from embeddings.client import EmbeddingClient, read_embedding_model_name
from hooks.manager import HookManager
from models.classifier_client import classify_intent_with_model
from models.clarification_planner import plan_clarification_with_model
from models.llm_client import ModelAnswerResult, call_chat_model
from rag.knowledge_base import load_knowledge_chunks
from rag.prompting import (
    build_citations,
    build_product_tool_rag_fallback,
    render_product_tool_rag_messages,
    render_rag_messages,
)
from rag.hybrid_retrieval import retrieve_hybrid_candidates
from rag.quality import is_low_confidence, run_rag_quality_check
from rag.product_joint import select_product_joint_hits
from rag.query_rewrite import normalize_query, rewrite_retrieval_query
from rag.reranker import RerankConfig, rerank_candidates
from rag.planning import is_realtime_business_query
from tools.contracts import TOOL_SPECS
from tools.planning import (
    build_clarification_plan,
    extract_order_id,
    extract_refund_request_id,
    extract_sku,
    is_product_tool_rag_query,
    post_tool_clarification,
    pre_tool_clarification,
    should_route_to_realtime_tool,
)
from tools.runtime_context import public_runtime_context
from tools.tool_calling import ToolCallingOutcome, ToolCallingService


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
        "refund_status_query",
        ["退款进度", "退款到哪", "退款到账", "退到哪"],
        "用户在查询已有退款申请的实时进度，必须交给只读业务工具核验。",
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

    message = normalize_query(user_message)
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

    refund_status = next(
        (item for item in active_evidence if item[0] == "refund_status_query"),
        None,
    )
    if refund_status:
        return IntentResult(
            intent="refund_status_query",
            source="rules",
            confidence=0.95,
            matched_keywords=refund_status[1],
            explanation=refund_status[3],
        )

    if extract_refund_request_id(message) and any(
        term in message for term in ["退款", "状态", "进度", "到账"]
    ):
        return IntentResult(
            intent="refund_status_query",
            source="rules",
            confidence=0.95,
            matched_keywords=["显式退款申请号"],
            explanation="用户提供了退款申请号并查询实时状态，规则高置信路由到只读退款工具。",
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

    if extract_order_id(message) and any(
        term in message for term in ["订单", "状态", "物流", "快递", "发货", "运单"]
    ):
        return IntentResult(
            intent="order_query",
            source="rules",
            confidence=0.95,
            matched_keywords=["显式订单号"],
            explanation="用户提供了订单号并查询实时状态，规则高置信路由到只读订单工具。",
        )

    if extract_sku(message) and any(
        term in message for term in ["库存", "价格", "多少钱", "还有货", "有没有货", "现价"]
    ):
        return IntentResult(
            intent="product_consult",
            source="rules",
            confidence=0.95,
            matched_keywords=["显式 SKU"],
            explanation="用户提供了 SKU 并查询实时价格或库存，规则高置信路由到只读商品工具。",
        )

    if is_product_tool_rag_query("product_consult", message):
        return IntentResult(
            intent="product_consult",
            source="rules",
            confidence=0.95,
            matched_keywords=["商品实时事实", "商品稳定知识"],
            explanation=(
                "用户同时询问商品实时价格或库存与稳定卖点或活动规则，"
                "路由到 Tool + RAG 联合回答。"
            ),
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
        "refund_status_query": "退款进度属于实时业务事实，需要通过只读退款工具核验。",
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


def build_realtime_business_gap_answer() -> str:
    """Keep per-user business state outside stable knowledge and its cache."""

    return (
        "这个问题涉及订单、物流、库存或退款进度等实时业务状态，"
        "稳定知识库无法核验，当前也还没有接入业务查询工具，因此不能编造结果。"
    )


def build_clarification_answer(clarification: ClarificationRequest) -> str:
    """Render a deterministic question without allowing the model to pick an option."""

    if not clarification.candidates:
        return clarification.message
    choices = "；".join(
        f"{candidate.label}（{candidate.hint}）"
        for candidate in clarification.candidates
    )
    return f"{clarification.message} 候选项：{choices}。"


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
        rerank_http_client: httpx.Client | None = None,
        rerank_config: RerankConfig | None = None,
        tool_calling_service: ToolCallingService | None = None,
        clarification_http_client: httpx.Client | None = None,
        clarification_api_key: str | None = None,
        clarification_base_url: str | None = None,
        clarification_model_name: str | None = None,
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
        self._rerank_http_client = rerank_http_client
        self._rerank_config = rerank_config
        self._tool_calling_service = tool_calling_service or ToolCallingService()
        self._clarification_http_client = clarification_http_client
        self._clarification_api_key = clarification_api_key
        self._clarification_base_url = clarification_base_url
        self._clarification_model_name = clarification_model_name

    def _finalize_with_hooks(
        self,
        response: ChatResponse,
        hooks: HookManager,
    ) -> ChatResponse:
        """Attach one bounded lifecycle summary to every public response path."""

        degradation = DegradationState.model_validate(
            response.session_state.get("degradation", {})
        )
        if response.degraded and not any(
            event.hook_type == "on_error" for event in hooks.events
        ):
            hooks.on_error(
                "agent_response",
                degradation.error_category,
                degradation.reason or "当前请求已进入安全降级路径。",
                degradation.retry_count + 1,
            )
        completion = hooks.on_completion(
            next_action=response.next_action,
            risk_level=response.risk_level,
            degradation=degradation,
        )
        state = dict(response.session_state)
        state["agent_version"] = "0.17.0"
        state["hooks"] = {
            "events": [event.model_dump() for event in hooks.events],
            "completion": completion.model_dump(),
            "full_trace_available": False,
            "hitl_approval_performed": False,
        }
        state["next_gap"] = (
            "工具链路已形成统一 Hooks 治理；下一步将工具、资源和 Prompt 接入标准化 MCP 来源。"
        )
        return response.model_copy(
            update={
                "hook_events": list(hooks.events),
                "hook_completion": completion,
                "session_state": state,
            }
        )

    def _high_risk_response(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        message_count: int,
    ) -> ChatResponse:
        """Block writes before RAG or tool planning and request human review."""

        degradation = high_risk_degradation()
        answer = fallback_message(degradation.error_category)
        messages = [
            {"role": "system", "content": "高风险写操作必须人工复核。"},
            {"role": "user", "content": request.user_message},
        ]
        cost_summary = build_cost_summary(messages, answer, None)
        event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "risk_level": "high",
            "next_action": "transfer_to_human",
            "cost_summary": cost_summary.model_dump(),
        }
        self._cost_events_by_session.setdefault(request.session_id, []).append(event)
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            citations=[],
            tool_calls=[],
            next_action="transfer_to_human",
            risk_level="high",
            needs_human_approval=True,
            degraded=True,
            cost_summary=cost_summary,
            reasoning_summary=[
                "后端在 RAG 和 Tool Calling 之前识别到退款、取消或赔付写请求。",
                "当前版本只开放只读工具，未执行任何写操作。",
                "请求已标记为高风险并转入人工确认边界。",
            ],
            session_state={
                "agent_version": "0.17.0",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": request.runtime_user_id,
                    "nickname": request.runtime_nickname,
                    "member_level": request.runtime_member_level,
                    "risk_level": request.runtime_risk_level,
                    "page_context": public_runtime_context(request),
                },
                "degradation": degradation.model_dump(),
                "risk_boundary": {
                    "blocked": True,
                    "risk_level": "high",
                    "needs_human_approval": True,
                    "write_executed": False,
                    "workflow_started": False,
                },
                "tool_calling": {
                    "create_agent": False,
                    "skip_reason": "high_risk_write_blocked",
                    "tool_call_count": 0,
                    "raw_tool_result_exposed": False,
                },
                "rag": {"status": "skipped_high_risk_write", "realtime_gap": False},
                "rag_quality": {"status": "skipped_high_risk_write"},
                "cost_log": {
                    "event_count": len(self._cost_events_by_session[request.session_id]),
                    "latest": event,
                },
                "next_gap": "已建立高风险写操作边界；下一步接入可恢复的 Workflow 与真实 HITL 审批。",
            },
        )

    def _chat_with_tools(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        message_count: int,
        hooks: HookManager,
    ) -> ChatResponse:
        """Run realtime reads without sending trusted identity to the planning model."""

        clarification_plan = build_clarification_plan(
            request,
            intent_result.intent,
        )
        clarification = None
        clarification_stage = None
        if clarification_plan.missing_required:
            clarification_plan = plan_clarification_with_model(
                request,
                clarification_plan,
                http_client=self._clarification_http_client,
                api_key=self._clarification_api_key,
                base_url=self._clarification_base_url,
                model=self._clarification_model_name,
            )
            clarification = pre_tool_clarification(request, clarification_plan)

        if clarification is not None:
            clarification_stage = "pre_tool"
            outcome = ToolCallingOutcome(
                answer=build_clarification_answer(clarification),
                tool_calls=[],
                state={
                    "create_agent": False,
                    "skip_reason": "clarification_required",
                    "available_tools": [
                        spec.model_dump() for spec in TOOL_SPECS.values()
                    ],
                    "message_types": [],
                    "answer_source": "structured_clarification",
                },
                used_model=clarification_plan.source == "model",
                model_name=clarification_plan.model_name,
            )
        else:
            outcome = self._tool_calling_service.run(
                request,
                intent_result.intent,
                clarification_plan,
                hooks=hooks,
            )
            clarification = post_tool_clarification(outcome.tool_calls)
            if clarification is not None:
                clarification_stage = "post_tool"
        response_answer = (
            build_clarification_answer(clarification)
            if clarification is not None
            else outcome.answer
        )
        degradation = degradation_from_tool_records(
            outcome.tool_calls,
            model_error=outcome.error if clarification is None else None,
        )
        if clarification is None and degradation.degraded:
            response_answer = fallback_message(degradation.error_category)
        if clarification is not None:
            next_action = "ask_clarification"
        elif outcome.tool_calls:
            next_action = outcome.tool_calls[-1].observation.next_action
        else:
            next_action = "fallback_answer"
        risk_level = "medium" if any(
            TOOL_SPECS[record.action.tool_name].risk_level == "medium"
            for record in outcome.tool_calls
            if record.action.tool_name in TOOL_SPECS
        ) else "low"
        public_messages = [
            {
                "role": "system",
                "content": "实时业务事实必须通过后端受控的只读工具核验。",
            },
            {"role": "user", "content": request.user_message},
        ]
        cost_summary = build_cost_summary(public_messages, response_answer, None)
        event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "tool_names": [
                record.action.tool_name for record in outcome.tool_calls
            ],
            "next_action": next_action,
            "error_category": degradation.error_category,
            "retry_count": degradation.retry_count,
            "cost_summary": cost_summary.model_dump(),
        }
        self._cost_events_by_session.setdefault(request.session_id, []).append(event)
        reasoning_summary = [
            "实时事实与稳定知识已分流，本轮不使用 RAG 猜测业务状态。",
            "规划模型只接收用户问题和只读工具 schema，不接收 runtime_user_id。",
            "后端校验工具名与参数，并从可信 Runtime Context 注入当前用户身份。",
            f"本轮得到 {len(outcome.tool_calls)} 条经过脱敏的 Action/Observation 记录。",
            "内部 ToolResult 已按字段白名单压缩，原始业务 payload 不进入模型或公开响应。",
            (
                f"本轮澄清阶段为 {clarification_stage}，模型不能替用户选择候选。"
                if clarification_stage
                else "本轮工具参数唯一且完整，不需要向用户澄清。"
            ),
        ]
        return ChatResponse(
            session_id=request.session_id,
            answer=response_answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            citations=[],
            tool_calls=outcome.tool_calls,
            clarification=clarification,
            next_action=next_action,
            risk_level=risk_level,
            needs_human_approval=False,
            degraded=degradation.degraded,
            cost_summary=cost_summary,
            reasoning_summary=reasoning_summary,
            session_state={
                "agent_version": "0.17.0",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": request.runtime_user_id,
                    "nickname": request.runtime_nickname,
                    "member_level": request.runtime_member_level,
                    "risk_level": request.runtime_risk_level,
                    "page_context": public_runtime_context(request),
                },
                "model_answer": {
                    "used_model": outcome.used_model,
                    "model_name": outcome.model_name,
                    "fallback_reason": outcome.error,
                    "source": outcome.state.get("answer_source"),
                },
                "degradation": degradation.model_dump(),
                "tool_calling": {
                    **outcome.state,
                    "clarification_plan": clarification_plan.model_dump(),
                    "clarification": (
                        clarification.model_dump() if clarification else None
                    ),
                    "clarification_stage": clarification_stage,
                    "tool_call_count": len(outcome.tool_calls),
                    "actions": [
                        record.action.model_dump() for record in outcome.tool_calls
                    ],
                    "observations": [
                        record.observation.model_dump()
                        for record in outcome.tool_calls
                    ],
                    "raw_tool_result_exposed": False,
                    "observation_compression": True,
                    "next_action": next_action,
                },
                "rag": {
                    "status": "skipped_realtime_tool_route",
                    "realtime_gap": False,
                },
                "rag_quality": {"status": "skipped_realtime_tool_route"},
                "cost_log": {
                    "event_count": len(
                        self._cost_events_by_session[request.session_id]
                    ),
                    "latest": event,
                },
                "next_gap": "已支持错误分类、只读超时有限重试和安全降级；下一步接入可恢复的 Workflow 与真实 HITL 审批。",
            },
        )

    def _chat_with_product_tool_rag(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        message_count: int,
        hooks: HookManager,
    ) -> ChatResponse:
        """Combine current product facts with stable, cited product knowledge."""

        tool_response = self._chat_with_tools(
            request,
            intent_result,
            message_count,
            hooks,
        )
        successful_observations = [
            record.observation
            for record in tool_response.tool_calls
            if record.observation.status == "success"
        ]
        if (
            tool_response.clarification is not None
            or tool_response.degraded
            or not successful_observations
        ):
            tool_response.session_state["tool_rag"] = {
                "mode": "product_tool_plus_rag",
                "status": "stopped_before_joint_answer",
                "answer_sources": ["tool"] if successful_observations else [],
                "joint_answer_complete": False,
                "reason": (
                    "clarification_required"
                    if tool_response.clarification is not None
                    else "tool_unavailable"
                ),
            }
            return tool_response

        rewrite = rewrite_retrieval_query(
            request.user_message,
            intent_result.intent,
        )
        retrieval_error: str | None = None
        retrieval_outcome = None
        rerank_outcome = None
        reliable_hits: list[KnowledgeHit] = []
        try:
            retrieval_outcome = retrieve_hybrid_candidates(
                rewrite,
                intent_result.intent,
                embedding_client=self._embedding_client,
            )
            rerank_outcome = rerank_candidates(
                rewrite.rewritten_query,
                retrieval_outcome.candidates,
                config=self._rerank_config,
                http_client=self._rerank_http_client,
            )
            reliable_hits = select_product_joint_hits(rerank_outcome.hits)
        except (
            RuntimeError,
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            retrieval_error = exc.__class__.__name__

        citations = build_citations(reliable_hits)
        fallback_answer = build_product_tool_rag_fallback(
            successful_observations,
            reliable_hits,
        )
        messages = render_product_tool_rag_messages(
            request.user_message,
            successful_observations,
            reliable_hits,
        )
        if reliable_hits:
            model_answer = call_chat_model(
                messages,
                fallback_answer=fallback_answer,
                http_client=self._answer_http_client,
                api_key=self._answer_api_key,
                base_url=self._answer_base_url,
                model=self._answer_model_name,
            )
            if model_answer.used_model and "[C" not in model_answer.answer:
                model_answer = ModelAnswerResult(
                    answer=fallback_answer,
                    model_name=model_answer.model_name,
                    fallback_reason="citation_marker_missing",
                    usage=model_answer.usage,
                )
        else:
            model_answer = ModelAnswerResult(
                answer=fallback_answer,
                fallback_reason=(
                    retrieval_error or "joint_rag_low_confidence"
                ),
            )

        degraded = retrieval_error is not None or (
            bool(reliable_hits) and not model_answer.used_model
        )
        degradation_category = (
            "system_error"
            if retrieval_error is not None
            else "model_unavailable"
            if degraded
            else "none"
        )
        cost_summary = build_cost_summary(
            messages,
            model_answer.answer,
            model_answer.usage,
        )
        event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "route": "product_tool_plus_rag",
            "tool_names": [
                record.action.tool_name for record in tool_response.tool_calls
            ],
            "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
            "cost_summary": cost_summary.model_dump(),
        }
        events = self._cost_events_by_session.setdefault(request.session_id, [])
        if events:
            events[-1] = event
        else:
            events.append(event)

        state = tool_response.session_state
        state["agent_version"] = "0.17.0"
        state["model_answer"] = model_answer.model_dump()
        state["degradation"] = {
            "degraded": degraded,
            "error_category": degradation_category,
            "retry_count": state.get("degradation", {}).get("retry_count", 0),
            "fallback_used": not model_answer.used_model,
            "reason": model_answer.fallback_reason,
        }
        state["tool_calling"]["joint_answer"] = True
        state["rag"] = {
            "status": (
                "unavailable"
                if retrieval_error
                else "completed_product_joint_route"
            ),
            "mode": "hybrid_rag_for_product_tool_joint_answer",
            "vector_search": retrieval_outcome is not None,
            "keyword_search": retrieval_outcome is not None,
            "rewrite": rewrite.model_dump(),
            "plan": (
                retrieval_outcome.plan.model_dump()
                if retrieval_outcome is not None
                else None
            ),
            "cache": (
                retrieval_outcome.cache if retrieval_outcome is not None else None
            ),
            "candidate_count": (
                len(retrieval_outcome.candidates)
                if retrieval_outcome is not None
                else 0
            ),
            "retrieved_count": len(reliable_hits),
            "citation_count": len(citations),
            "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
            "rerank_mode": rerank_outcome.mode if rerank_outcome else "skipped",
            "rerank_model": rerank_outcome.model if rerank_outcome else None,
            "rerank_error": rerank_outcome.error if rerank_outcome else retrieval_error,
            "retrieval_error": retrieval_error,
        }
        state["rag_quality"] = {"status": "skipped_product_joint_route"}
        state["tool_rag"] = {
            "mode": "product_tool_plus_rag",
            "status": "completed" if citations else "partial_tool_only",
            "answer_sources": ["tool", "rag"] if citations else ["tool"],
            "current_fact_source": "tool_observation",
            "stable_knowledge_source": "rag_citations" if citations else None,
            "tool_names": [
                record.action.tool_name for record in tool_response.tool_calls
            ],
            "citation_chunk_ids": [citation.chunk_id for citation in citations],
            "joint_answer_complete": bool(successful_observations and citations),
            "source_boundary": {
                "current_price_inventory": "tool",
                "product_and_promotion_knowledge": "rag",
            },
        }
        state["cost_log"] = {"event_count": len(events), "latest": event}
        state["next_gap"] = (
            "已支持商品 Tool + RAG 联合回答；下一步治理重复的工具校验、错误处理与日志链路。"
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            citations=citations,
            tool_calls=tool_response.tool_calls,
            clarification=None,
            next_action=tool_response.next_action,
            risk_level=tool_response.risk_level,
            needs_human_approval=False,
            degraded=degraded,
            cost_summary=cost_summary,
            reasoning_summary=[
                "商品当前价格、库存和活动状态来自受控只读工具 Observation。",
                "商品卖点与平台活动规则来自 Hybrid RAG 的可靠命中并生成 citations。",
                "联合回答只接收脱敏 Observation 和可靠知识块，不接收原始业务 payload。",
                "平台规则不代表当前 SKU 必然享有优惠，最终优惠以结算页为准。",
            ],
            session_state=state,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Classify, retrieve vectors, generate a grounded answer, and cite hits."""

        self._message_count_by_session[request.session_id] = (
            self._message_count_by_session.get(request.session_id, 0) + 1
        )
        message_count = self._message_count_by_session[request.session_id]
        hooks = HookManager()
        intent_result = classify_intent(
            request.user_message,
            http_client=self._classifier_http_client,
            api_key=self._classifier_api_key,
            base_url=self._classifier_base_url,
            model=self._classifier_model_name,
        )
        if is_high_risk_write_request(request.user_message):
            return self._finalize_with_hooks(
                self._high_risk_response(request, intent_result, message_count),
                hooks,
            )
        if is_product_tool_rag_query(
            intent_result.intent,
            request.user_message,
        ):
            return self._finalize_with_hooks(
                self._chat_with_product_tool_rag(
                    request,
                    intent_result,
                    message_count,
                    hooks,
                ),
                hooks,
            )
        if should_route_to_realtime_tool(
            intent_result.intent,
            request.user_message,
        ):
            return self._finalize_with_hooks(
                self._chat_with_tools(
                    request,
                    intent_result,
                    message_count,
                    hooks,
                ),
                hooks,
            )

        rewrite = rewrite_retrieval_query(
            request.user_message,
            intent_result.intent,
        )
        original_candidates: list[KnowledgeHit] = []
        rewritten_candidates: list[KnowledgeHit] = []
        keyword_candidates: list[KnowledgeHit] = []
        candidates: list[KnowledgeHit] = []
        retrieval_plan = None
        retrieval_index = None
        retrieval_cache: dict = {
            "cacheable": False,
            "scope": "hybrid_candidates_only",
            "reason": "普通聊天不执行知识检索。",
            "cache_hit": False,
            "cache_key": None,
            "entry_count": 0,
            "embedding_identity_hash": None,
        }
        reranked_hits: list[KnowledgeHit] = []
        reliable_hits: list[KnowledgeHit] = []
        retrieval_error: str | None = None
        rerank_mode = "skipped_general_chat"
        rerank_model: str | None = None
        rerank_error: str | None = None
        answer_path = "general_chat"
        low_confidence = False
        realtime_gap = False
        if intent_result.intent != "general_chat":
            try:
                retrieval_outcome = retrieve_hybrid_candidates(
                    rewrite,
                    intent_result.intent,
                    embedding_client=self._embedding_client,
                )
                retrieval_plan = retrieval_outcome.plan
                retrieval_index = retrieval_outcome.index
                retrieval_cache = retrieval_outcome.cache
                original_candidates = retrieval_outcome.original_vector_hits
                rewritten_candidates = retrieval_outcome.rewritten_vector_hits
                keyword_candidates = retrieval_outcome.keyword_hits
                candidates = retrieval_outcome.candidates
                rerank_outcome = rerank_candidates(
                    rewrite.rewritten_query,
                    candidates,
                    config=self._rerank_config,
                    http_client=self._rerank_http_client,
                )
                reranked_hits = rerank_outcome.hits
                rerank_mode = rerank_outcome.mode
                rerank_model = rerank_outcome.model
                rerank_error = rerank_outcome.error
                realtime_gap = is_realtime_business_query(request.user_message)
                low_confidence = realtime_gap or is_low_confidence(reranked_hits)
                reliable_hits = [] if low_confidence else reranked_hits[:FINAL_TOP_K]
                if realtime_gap:
                    answer_path = "realtime_business_tool_required"
                else:
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
                rerank_mode = "skipped_retrieval_unavailable"
                low_confidence = True
                answer_path = "retrieval_unavailable"

        citations = build_citations(reliable_hits)
        messages = render_rag_messages(
            request.user_message,
            intent_result,
            rewrite,
            reliable_hits,
        )
        if intent_result.intent == "general_chat":
            fallback_answer = build_answer(intent_result)
        elif realtime_gap:
            fallback_answer = build_realtime_business_gap_answer()
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
        degraded = retrieval_error is not None or (
            bool(reliable_hits) and not model_answer.used_model
        )
        degradation_category = (
            "system_error"
            if retrieval_error is not None
            else "model_unavailable"
            if degraded
            else "none"
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
        chunks = (
            list(retrieval_index.chunks_by_id.values())
            if retrieval_index is not None
            else load_knowledge_chunks()
        )
        top_score = reranked_hits[0].score if reranked_hits else 0.0
        if intent_result.intent == "general_chat":
            confidence_level = "not_applicable"
            low_confidence_action = "general_chat_without_citations"
        elif retrieval_error:
            confidence_level = "unavailable"
            low_confidence_action = "clarify_or_handoff"
        elif realtime_gap:
            confidence_level = "not_applicable"
            low_confidence_action = "business_tool_required"
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
            f"系统保留用户原话，生成检索改写与 pre-retrieval 场景计划；三路合并后得到 {len(candidates)} 个候选。",
            f"知识索引版本为 {retrieval_index.version if retrieval_index else 'not_applicable'}；稳定检索缓存命中为 {retrieval_cache['cache_hit']}。",
            f"候选经过 {rerank_mode} 重排，再用 {LOW_CONFIDENCE_THRESHOLD} 门槛选出 {len(reliable_hits)} 个可靠命中。",
            f"最终可靠知识生成 {len(citations)} 条 citation；可信 Runtime Context 未进入外部检索或回答请求。",
            f"本轮 token 来源为 {cost_summary.token_source}，总 token 为 {cost_summary.total_tokens}。",
        ]
        session_state = {
            "agent_version": "0.17.0",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": request.runtime_context or {},
            },
            "model_answer": model_answer.model_dump(),
            "degradation": {
                "degraded": degraded,
                "error_category": degradation_category,
                "retry_count": 0,
                "fallback_used": degraded,
                "reason": retrieval_error or model_answer.fallback_reason if degraded else None,
            },
            "rag": {
                "mode": "hybrid_rag_with_versioned_index_cache",
                "retrieval_strategy": "versioned_index_then_cached_hybrid_candidates_then_rerank",
                "vector_search": True,
                "keyword_search": True,
                "embedding_model": read_embedding_model_name(self._embedding_client),
                "candidate_k": CANDIDATE_K,
                "hybrid_candidate_k": HYBRID_CANDIDATE_K,
                "final_top_k": FINAL_TOP_K,
                "score_threshold": RETRIEVAL_SCORE_THRESHOLD,
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "document_count": len({chunk.source_path for chunk in chunks}),
                "chunk_count": len(chunks),
                "index": (
                    {
                        "version": retrieval_index.version,
                        "fingerprint": retrieval_index.fingerprint,
                        "chunk_count": retrieval_index.chunk_count,
                        "document_count": retrieval_index.document_count,
                        "inverted_term_count": len(retrieval_index.inverted_index),
                    }
                    if retrieval_index
                    else None
                ),
                "cache": retrieval_cache,
                "realtime_gap": realtime_gap,
                "rewrite": rewrite.model_dump(),
                "plan": retrieval_plan.model_dump() if retrieval_plan else None,
                "original_candidate_count": len(original_candidates),
                "rewritten_candidate_count": len(rewritten_candidates),
                "keyword_candidate_count": len(keyword_candidates),
                "candidate_count": len(candidates),
                "retrieved_count": len(reliable_hits),
                "citation_count": len(citations),
                "confidence_level": confidence_level,
                "top_score": top_score,
                "low_confidence_action": low_confidence_action,
                "answer_path": answer_path,
                "initial_top_chunk_id": (
                    original_candidates[0].chunk.chunk_id
                    if original_candidates
                    else None
                ),
                "candidate_chunk_ids": [hit.chunk.chunk_id for hit in candidates],
                "original_vector_chunk_ids": [
                    hit.chunk.chunk_id for hit in original_candidates
                ],
                "rewritten_vector_chunk_ids": [
                    hit.chunk.chunk_id for hit in rewritten_candidates
                ],
                "keyword_chunk_ids": [hit.chunk.chunk_id for hit in keyword_candidates],
                "reranked_chunk_ids": [hit.chunk.chunk_id for hit in reranked_hits],
                "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable_hits],
                "rerank_mode": rerank_mode,
                "rerank_model": rerank_model,
                "rerank_error": rerank_error,
                "rerank_reasons": {
                    hit.chunk.chunk_id: hit.rerank_reasons for hit in reranked_hits
                },
                "scores": {
                    hit.chunk.chunk_id: {
                        "vector": hit.vector_score,
                        "keyword": hit.keyword_score,
                        "rerank": hit.rerank_score,
                        "final": hit.score,
                        "sources": hit.retrieval_sources,
                        "matched_keywords": hit.matched_keywords,
                    }
                    for hit in reranked_hits
                },
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
            "next_gap": "已支持错误分类、只读超时有限重试和安全降级；下一步接入可恢复的 Workflow 与真实 HITL 审批。",
        }
        return self._finalize_with_hooks(
            ChatResponse(
                session_id=request.session_id,
                answer=model_answer.answer,
                intent=intent_result.intent,
                intent_result=intent_result,
                citations=citations,
                tool_calls=[],
                next_action="answer_user",
                risk_level="low",
                needs_human_approval=False,
                degraded=degraded,
                cost_summary=cost_summary,
                reasoning_summary=reasoning_summary,
                session_state=session_state,
            ),
            hooks,
        )
