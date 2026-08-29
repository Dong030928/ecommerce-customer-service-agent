"""Retrieval-only query normalization and intent-aware expansion."""

from __future__ import annotations

from api.schemas import Intent, QueryRewrite
from config.settings import NORMALIZATION_RULES


def normalize_query(text: str) -> str:
    """Align conversational expressions with terms commonly used in the corpus."""

    normalized = text.lower()
    for source, target, _reason in NORMALIZATION_RULES:
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def describe_normalization(original_query: str) -> list[str]:
    """Return public-safe explanations for matched normalization rules."""

    query = original_query.lower()
    return [
        reason for source, _target, reason in NORMALIZATION_RULES if source in query
    ]


def add_rewrite_terms(target: list[str], terms: list[str]) -> None:
    """Append unique retrieval terms while preserving their declared order."""

    for term in terms:
        if term not in target:
            target.append(term)


def build_rewrite_reason(
    normalization_reasons: list[str],
    rewrite_reasons: list[str],
) -> str:
    reasons = [*normalization_reasons, *rewrite_reasons]
    if reasons:
        return "；".join(reasons) + "。"
    return "未命中归一化或补词规则，保留原问题直接检索，再对候选知识进行重排。"


def rewrite_retrieval_query(user_message: str, intent: Intent) -> QueryRewrite:
    """Build a retrieval query without reading trusted runtime identity fields."""

    normalized = normalize_query(user_message)
    added_terms: list[str] = []
    rewrite_reasons: list[str] = []

    if intent == "promotion_consult" and "耳机" in normalized:
        add_rewrite_terms(added_terms, ["当前", "2026", "春季音频节"])
        rewrite_reasons.append("耳机促销咨询补齐当前活动时间和活动名")
        add_rewrite_terms(added_terms, ["降噪耳机"])
        rewrite_reasons.append("补齐知识库中的具体商品类目")
        add_rewrite_terms(added_terms, ["会员价", "优惠券", "叠加", "结算页"])
        rewrite_reasons.append("补齐优惠叠加与结算页边界，增强重排信号")
    elif intent == "refund_request":
        add_rewrite_terms(added_terms, ["售后规则", "签收时间", "退货条件", "凭证"])
        rewrite_reasons.append("售后意图补齐签收时间、退货条件和凭证要求")

    rewritten_query = (
        " ".join(part for part in [normalized, *added_terms] if part).strip()
        or user_message
    )
    return QueryRewrite(
        original_query=user_message,
        rewritten_query=rewritten_query,
        applied=rewritten_query != user_message,
        added_terms=added_terms,
        reason=build_rewrite_reason(
            describe_normalization(user_message),
            rewrite_reasons,
        ),
    )
