"""Select a bounded model context without losing protected current facts."""

from __future__ import annotations

from api.schemas import (
    CompressionReport,
    ContextBuildReport,
    ContextCandidate,
    HistoryMessage,
    SourceType,
)
from config.settings import MAX_CONTEXT_TOKENS, RECENT_WINDOW_SIZE
from cost.observer import estimate_tokens
from tools.planning import extract_order_id


PROTECTED_SOURCES: set[SourceType] = {
    "runtime_context",
    "tool_observation",
    "workflow_state",
}
SOURCE_RELEVANCE: dict[SourceType, int] = {
    "runtime_context": 100,
    "workflow_state": 100,
    "tool_observation": 95,
    "rag_snippet": 90,
    "session_memory": 70,
    "user_message": 100,
    "history_message": 35,
}


class ContextCompressor:
    """Apply protected-source, relevance, and Sliding Window selection rules."""

    def __init__(
        self,
        *,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        recent_window_size: int = RECENT_WINDOW_SIZE,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._recent_window_size = recent_window_size

    def compress(
        self,
        report: ContextBuildReport,
        *,
        history_messages: list[HistoryMessage],
        current_message: str,
    ) -> CompressionReport:
        """Keep protected facts first, then select optional items by relevance."""

        candidates = self._context_candidates(report)
        candidates.extend(
            self._history_candidates(history_messages, current_message=current_message)
        )
        before = sum(item.token_estimate for item in candidates)
        protected = [item for item in candidates if item.protected]
        optional = [item for item in candidates if not item.protected]
        kept = list(protected)
        budget = self._max_context_tokens - sum(
            item.token_estimate for item in protected
        )

        for item in sorted(optional, key=self._sort_key, reverse=True):
            if item.token_estimate <= budget or item.relevance_score >= 85:
                kept.append(item)
                budget -= item.token_estimate

        kept_ids = {item.item_id for item in kept}
        dropped = [item for item in candidates if item.item_id not in kept_ids]
        compressed_summary = self._summarize_dropped(dropped)
        if compressed_summary:
            kept.append(
                ContextCandidate(
                    item_id="compressed-history-summary",
                    source_type="history_message",
                    trust_level="untrusted",
                    content=compressed_summary,
                    token_estimate=estimate_tokens(compressed_summary),
                    relevance_score=60,
                    keep_reason="旧历史压缩为公开摘要，避免长上下文淹没当前事实。",
                )
            )

        return CompressionReport(
            max_context_tokens=self._max_context_tokens,
            recent_window_size=self._recent_window_size,
            input_items=candidates,
            kept_items=kept,
            model_context=[item.content for item in kept],
            compressed_summary=compressed_summary,
            dropped_items=dropped,
            token_estimate_before=before,
            token_estimate_after=sum(item.token_estimate for item in kept),
            lost_in_middle_guardrails=[
                "当前用户消息、Runtime Context、Tool Observation 和 Workflow State 标记为 protected。",
                "最近消息通过 Sliding Window 保留。",
                "中间历史命中当前订单号时按相关性保留。",
                "旧低相关历史只进入压缩摘要，不直接进入模型上下文候选。",
                "服务端 Workflow checkpoint 独立保存，不依赖聊天历史或压缩摘要。",
            ],
        )

    @staticmethod
    def _context_candidates(report: ContextBuildReport) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for item in report.selected_items:
            protected = (
                item.source_type in PROTECTED_SOURCES
                or item.item_id == "user-message"
            )
            relevance = SOURCE_RELEVANCE[item.source_type]
            if item.source_type == "session_memory" and item.facts.get(
                "last_order_id"
            ):
                relevance = 88
            candidates.append(
                ContextCandidate(
                    item_id=item.item_id,
                    source_type=item.source_type,
                    trust_level=item.trust_level,
                    content=item.content,
                    token_estimate=estimate_tokens(item.content),
                    relevance_score=relevance,
                    protected=protected,
                    keep_reason=(
                        "可信当前事实或流程边界不能被历史消息裁掉。"
                        if protected
                        else item.decision
                    ),
                )
            )
        return candidates

    def _history_candidates(
        self,
        history_messages: list[HistoryMessage],
        *,
        current_message: str,
    ) -> list[ContextCandidate]:
        current_order_id = extract_order_id(current_message)
        total = len(history_messages)
        candidates: list[ContextCandidate] = []
        for index, message in enumerate(history_messages):
            is_recent = index >= max(0, total - self._recent_window_size)
            history_order_id = extract_order_id(message.content)
            shares_order = bool(
                current_order_id
                and history_order_id
                and current_order_id == history_order_id
            )
            relevance = 82 if is_recent else 35
            keep_reason = "最近 Sliding Window 消息。" if is_recent else None
            if shares_order:
                relevance = 92
                keep_reason = "命中当前订单号，防止 Lost in the Middle。"
            content = f"{message.role}: {message.content}"
            candidates.append(
                ContextCandidate(
                    item_id=f"history-{index}",
                    source_type="history_message",
                    trust_level="untrusted",
                    content=content,
                    token_estimate=estimate_tokens(content),
                    relevance_score=relevance,
                    protected=is_recent,
                    keep_reason=keep_reason,
                )
            )
        return candidates

    @staticmethod
    def _sort_key(candidate: ContextCandidate) -> tuple[int, int]:
        if candidate.item_id.startswith("history-"):
            return candidate.relevance_score, int(
                candidate.item_id.removeprefix("history-")
            )
        return candidate.relevance_score, 10_000

    @staticmethod
    def _summarize_dropped(dropped: list[ContextCandidate]) -> str:
        if not dropped:
            return ""
        order_ids: list[str] = []
        for item in dropped:
            order_id = extract_order_id(item.content)
            if order_id and order_id not in order_ids:
                order_ids.append(order_id)
        order_text = f"；旧历史提到过订单 {', '.join(order_ids)}" if order_ids else ""
        return f"压缩摘要：已折叠 {len(dropped)} 条低相关上下文{order_text}。"
