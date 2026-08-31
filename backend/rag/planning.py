"""Pre-retrieval scene routing and lexical-query planning."""

from __future__ import annotations

import re

from api.schemas import Intent, QueryRewrite, RetrievalPlan, RetrievalScene
from rag.query_rewrite import normalize_query


SCENE_DOMAINS: dict[RetrievalScene, list[str]] = {
    "promotion": ["promotion", "member"],
    "after_sale": ["after_sale"],
    "shipping": ["shipping"],
    "product": ["product", "promotion"],
    "order": ["order", "shipping"],
    "complaint": ["complaint", "after_sale"],
    "unknown": [
        "promotion",
        "member",
        "after_sale",
        "shipping",
        "product",
        "order",
        "payment",
        "complaint",
    ],
}

SCENE_TERMS: dict[RetrievalScene, list[str]] = {
    "promotion": [
        "当前",
        "2026",
        "春季音频节",
        "金卡",
        "降噪",
        "耳机",
        "活动",
        "会员价",
        "优惠券",
        "叠加",
        "结算页",
    ],
    "after_sale": [
        "售后",
        "退货",
        "退款",
        "无理由",
        "签收",
        "7天",
        "八天",
        "配件",
        "赠品",
        "包装",
        "包装盒",
        "压坏",
        "缺失",
        "凭证",
    ],
    "shipping": ["物流", "快递", "发货", "运单", "时效", "现货", "预售", "48小时"],
    "product": ["商品", "耳机", "充电器", "音箱", "推荐"],
    "order": ["订单", "取消", "地址", "支付"],
    "complaint": ["投诉", "举报", "赔偿", "人工"],
    "unknown": [],
}

SCENE_SIGNALS: list[tuple[RetrievalScene, list[str]]] = [
    (
        "after_sale",
        ["配件", "赠品", "包装", "无理由", "退货", "退款", "质量问题", "售后"],
    ),
    ("shipping", ["物流", "快递", "发货", "运单", "配送"]),
    ("promotion", ["优惠", "活动", "会员价", "优惠券", "满减", "叠加", "秒杀"]),
    ("complaint", ["投诉", "举报", "赔偿", "曝光", "315"]),
    ("order", ["订单", "取消订单", "修改地址"]),
    ("product", ["耳机", "充电器", "音箱", "推荐", "哪个好"]),
]

INTENT_SCENES: dict[Intent, RetrievalScene] = {
    "promotion_consult": "promotion",
    "refund_request": "after_sale",
    "refund_status_query": "after_sale",
    "order_query": "order",
    "complaint": "complaint",
    "product_consult": "product",
    "general_chat": "unknown",
    "unknown": "unknown",
}


def detect_retrieval_scene(query: str, intent: Intent) -> RetrievalScene:
    """Select a stable knowledge scene from query evidence, then intent."""

    normalized = normalize_query(query)
    # Structured safety-sensitive intents take priority over overlapping nouns.
    if intent == "complaint":
        return "complaint"
    if intent == "refund_request":
        return "after_sale"
    for scene, signals in SCENE_SIGNALS:
        if any(signal in normalized for signal in signals):
            return scene
    return INTENT_SCENES[intent]


def collect_keyword_terms(query: str, scene: RetrievalScene) -> list[str]:
    """Collect explicit long-tail terms plus bounded scene vocabulary."""

    normalized = normalize_query(query)
    explicit_terms = [
        term
        for term in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", normalized)
        if len(term) >= 2
    ]
    matched_scene_terms = [term for term in SCENE_TERMS[scene] if term in normalized]
    return list(dict.fromkeys([*matched_scene_terms, *explicit_terms]))


def build_retrieval_plan(
    rewrite: QueryRewrite,
    intent: Intent,
) -> RetrievalPlan:
    """Build a retrieval-only plan without reading trusted runtime context."""

    scene = detect_retrieval_scene(rewrite.rewritten_query, intent)
    allowed_domains = SCENE_DOMAINS[scene]
    keyword_terms = collect_keyword_terms(rewrite.rewritten_query, scene)
    if scene == "unknown":
        reason = "未识别到稳定知识场景，保留全领域候选并仅使用问题中的显式关键词。"
    else:
        reason = (
            f"识别为 {scene} 检索场景，只在 {', '.join(allowed_domains)} 领域召回，"
            "再合并向量与关键词证据。"
        )
    return RetrievalPlan(
        original_query=rewrite.original_query,
        rewritten_query=rewrite.rewritten_query,
        scene=scene,
        allowed_domains=allowed_domains,
        keyword_terms=keyword_terms,
        reason=reason,
    )


def is_realtime_business_query(query: str) -> bool:
    """Detect per-user business state that stable knowledge cannot answer or cache."""

    normalized = normalize_query(query)
    realtime_patterns = [
        "我的订单",
        "订单状态",
        "订单到哪",
        "快递到哪",
        "物流到哪",
        "物流状态",
        "发货了吗",
        "退款进度",
        "退款到哪",
        "退款到账",
        "库存还有",
        "有没有库存",
        "还有货吗",
    ]
    return any(pattern in normalized for pattern in realtime_patterns)
