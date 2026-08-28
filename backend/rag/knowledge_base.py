"""Markdown parsing, metadata preservation, and stable knowledge chunking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from api.schemas import KnowledgeChunk, KnowledgeSection, SourceDocument
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR


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
    """Split YAML-like front matter from a Markdown body."""

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
    """Parse section metadata stored in an HTML comment."""

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
    """Load all repository-local Markdown knowledge documents."""

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
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def parse_sections(document: SourceDocument) -> list[KnowledgeSection]:
    """Parse Markdown H2 sections while preserving source metadata."""

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
        keywords = _as_string_list(
            current_metadata.get("keywords") or document.metadata.get("tags") or []
        )
        chunk_id = current_metadata.get("chunk_id")
        effective_status = str(
            current_metadata.get("effective_status")
            or document.metadata.get("effective_status")
            or "active"
        )
        metadata = {
            **document.metadata,
            **current_metadata,
            "source_path": document.source_path,
            "document_title": document.title,
            "section": current_title,
            "section_index": section_index,
        }
        sections.append(
            KnowledgeSection(
                source_path=document.source_path,
                document_title=document.title,
                section_index=section_index,
                section=current_title,
                chunk_id=str(chunk_id) if chunk_id else None,
                keywords=keywords,
                effective_status=effective_status,
                text=text,
                metadata=metadata,
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


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split normalized section text with bounded overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size。")
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start += step
    return chunks


def build_knowledge_chunks(
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[KnowledgeChunk]:
    """Build stable, source-aware chunks from all knowledge documents."""

    chunks: list[KnowledgeChunk] = []
    for document in load_source_documents():
        for section in parse_sections(document):
            section_chunks = split_into_chunks(section.text, chunk_size, overlap)
            for chunk_index, chunk_text in enumerate(section_chunks, start=1):
                base_chunk_id = section.chunk_id or (
                    f"{Path(section.source_path).stem}-s{section.section_index}"
                )
                chunk_id = (
                    base_chunk_id
                    if len(section_chunks) == 1
                    else f"{base_chunk_id}-c{chunk_index}"
                )
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        document_title=section.document_title,
                        source_path=section.source_path,
                        section=section.section,
                        keywords=section.keywords,
                        effective_status=section.effective_status,
                        text=chunk_text,
                        metadata={
                            **section.metadata,
                            "chunk_index": chunk_index,
                            "chunk_count": len(section_chunks),
                        },
                    )
                )
    return chunks


def load_knowledge_chunks() -> list[KnowledgeChunk]:
    """Build the current knowledge chunk collection from Markdown sources."""

    return build_knowledge_chunks()


def query_asks_for_history(query: str) -> bool:
    lowered_query = query.lower()
    return any(term in lowered_query for term in ["历史", "复盘", "双11", "2024"])


def should_include_chunk_for_query(
    chunk: KnowledgeChunk,
    asks_for_history: bool,
) -> bool:
    """Exclude historical or scheduled content unless history is explicit."""

    return chunk.effective_status == "active" or asks_for_history
