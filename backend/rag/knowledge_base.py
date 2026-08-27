"""Basic RAG pipeline using Markdown metadata and keyword relevance.

This version intentionally does not use embeddings or a vector database. It
establishes the retrieve-before-answer boundary with an observable baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from api.schemas import (
    ChatRequest,
    Intent,
    IntentResult,
    KnowledgeHit,
    KnowledgeSection,
    KnowledgeSnippet,
    SourceDocument,
)
from config.settings import KNOWLEDGE_DIR, RAG_TOP_K


def parse_metadata_value(value: str) -> Any:
    """Parse simple scalar and list values used by the Markdown corpus."""

    parsed_value = value.strip()
    if parsed_value.startswith("[") and parsed_value.endswith("]"):
        return [
            item.strip().strip("\"'")
            for item in parsed_value[1:-1].split(",")
            if item.strip()
        ]
    if "," in parsed_value:
        return [
            item.strip().strip("\"'")
            for item in parsed_value.split(",")
            if item.strip()
        ]
    return parsed_value


def parse_front_matter(markdown: str) -> tuple[dict[str, Any], str]:
    """Split YAML-like front matter from the Markdown body."""

    if not markdown.startswith("---"):
        return {}, markdown
    end_marker = markdown.find("\n---", 3)
    if end_marker == -1:
        return {}, markdown

    raw_metadata = markdown[3:end_marker].strip()
    body = markdown[end_marker + 4 :].lstrip()
    metadata: dict[str, Any] = {}
    for raw_line in raw_metadata.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_metadata_value(value)
    return metadata, body


def parse_section_metadata(line: str) -> dict[str, Any]:
    """Parse metadata stored in a Markdown HTML comment."""

    stripped = line.strip()
    if not stripped.startswith("<!--") or not stripped.endswith("-->"):
        return {}
    raw_metadata = stripped.removeprefix("<!--").removesuffix("-->").strip()
    metadata: dict[str, Any] = {}
    for part in raw_metadata.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        metadata[key.strip()] = parse_metadata_value(value)
    return metadata


def load_source_documents() -> list[SourceDocument]:
    """Load the repository-local Markdown knowledge corpus."""

    documents: list[SourceDocument] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append(
            SourceDocument(
                source_path=path.name,
                title=str(metadata.get("title") or path.stem),
                metadata=metadata,
                body=body,
            )
        )
    return documents


def _as_string_list(value: Any) -> list[str]:
    """Normalize metadata that may be a scalar or list."""

    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def parse_sections(document: SourceDocument) -> list[KnowledgeSection]:
    """Split a source document into metadata-bearing business sections."""

    sections: list[KnowledgeSection] = []
    current_title = document.title
    current_lines: list[str] = []
    current_metadata: dict[str, Any] = {}
    section_index = 0

    def flush_section() -> None:
        nonlocal section_index, current_lines, current_title, current_metadata
        text = "\n".join(line for line in current_lines if line.strip()).strip()
        if not text:
            current_lines = []
            current_metadata = {}
            return
        section_index += 1
        raw_keywords = current_metadata.get("keywords") or document.metadata.get("tags") or []
        snippet_id = current_metadata.get("chunk_id")
        sections.append(
            KnowledgeSection(
                source_path=document.source_path,
                document_title=document.title,
                section_index=section_index,
                section=current_title,
                snippet_id=str(snippet_id) if snippet_id else None,
                keywords=_as_string_list(raw_keywords),
                effective_status=str(
                    current_metadata.get("effective_status")
                    or document.metadata.get("effective_status")
                    or "active"
                ),
                text=text,
            )
        )
        current_lines = []
        current_metadata = {}

    for raw_line in document.body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            flush_section()
            current_title = line.removeprefix("## ").strip()
            continue
        parsed_metadata = parse_section_metadata(line)
        if parsed_metadata:
            current_metadata.update(parsed_metadata)
            continue
        current_lines.append(line)
    flush_section()
    return sections


def topic_from_document(document: SourceDocument) -> Intent:
    """Map a knowledge domain to the existing coarse intent contract."""

    domain = str(document.metadata.get("domain") or "")
    return {
        "promotion": "promotion_consult",
        "after_sale": "refund_request",
        "shipping": "order_query",
        "order": "order_query",
        "product": "product_consult",
        "complaint": "complaint",
    }.get(domain, "unknown")


def load_knowledge_snippets() -> list[KnowledgeSnippet]:
    """Convert all Markdown sections into retrievable snippets."""

    snippets: list[KnowledgeSnippet] = []
    for document in load_source_documents():
        topic = topic_from_document(document)
        for section in parse_sections(document):
            snippet_id = section.snippet_id or (
                f"{Path(section.source_path).stem}-s{section.section_index}"
            )
            snippets.append(
                KnowledgeSnippet(
                    snippet_id=snippet_id,
                    title=section.section,
                    topic=topic,
                    source_path=section.source_path,
                    keywords=section.keywords,
                    effective_status=section.effective_status,
                    text=section.text,
                )
            )
    return snippets


def first_matched_keywords(message: str, keywords: list[str]) -> list[str]:
    """Return snippet keywords contained in the normalized user message."""

    return [keyword for keyword in keywords if keyword.lower() in message]


def retrieve_relevant_knowledge(
    user_message: str,
    intent: Intent,
    *,
    snippets: list[KnowledgeSnippet] | None = None,
    top_k: int = RAG_TOP_K,
) -> list[KnowledgeHit]:
    """Rank active snippets by keyword overlap plus a coarse-intent boost."""

    message = user_message.lower()
    asks_for_history = any(word in message for word in ["历史", "复盘", "双11", "2024"])
    hits: list[KnowledgeHit] = []
    for snippet in snippets if snippets is not None else load_knowledge_snippets():
        if snippet.effective_status != "active" and not asks_for_history:
            continue
        matched = first_matched_keywords(message, snippet.keywords)
        if not matched:
            continue
        keyword_score = len(matched) / max(len(snippet.keywords), 1)
        topic_boost = 0.2 if snippet.topic == intent else 0.0
        hits.append(
            KnowledgeHit(
                snippet=snippet,
                score=round(min(1.0, keyword_score + topic_boost), 3),
                matched_keywords=matched,
            )
        )
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def render_rag_messages(
    request: ChatRequest,
    intent_result: IntentResult,
    hits: list[KnowledgeHit],
) -> list[dict[str, str]]:
    """Render only retrieved knowledge without exposing trusted identity values."""

    retrieved_context = "\n\n".join(
        f"[{hit.snippet.snippet_id} | score={hit.score}]\n{hit.snippet.text}"
        for hit in hits
    ) or "当前问题没有命中相关知识。回答时要说明没有可靠依据，不能编造规则。"
    system_message = (
        "你是电商平台的客服 Agent。当前版本使用基础 RAG：只使用本轮检索到的相关知识，"
        "不得用模型常识补写平台规则。如果检索结果不足，必须承认没有可靠依据。"
    )
    user_message = (
        "平台运行时事实说明：\n"
        "- 用户身份、昵称、会员等级和风险等级由平台在本地可信上下文中维护，"
        "本轮不向外部模型披露具体值。\n"
        "- 如果回答依赖未提供给模型的运行时事实，应说明需要由平台系统核实。\n"
        "\n"
        f"粗意图：{intent_result.intent}\n"
        f"粗意图说明：{intent_result.explanation}\n"
        "\n"
        "本轮检索到的相关知识：\n"
        f"{retrieved_context}\n"
        "\n"
        "用户原话：\n"
        f"{request.user_message}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
