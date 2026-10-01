"""Deterministic, paragraph-aware parent/child splitting with source offsets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re

from api.schemas import KnowledgeChunk, KnowledgeParent, KnowledgeSection, SourceSpan


@dataclass(frozen=True)
class ChunkingConfig:
    parent_size: int = 1600
    parent_overlap: int = 160
    child_size: int = 420
    child_overlap: int = 80

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("分块配置必须是整数。")
        for size, overlap in (
            (self.parent_size, self.parent_overlap),
            (self.child_size, self.child_overlap),
        ):
            if size <= 0 or not 0 <= overlap < size:
                raise ValueError("分块 size 必须为正，overlap 必须在 [0, size) 内。")
        if self.parent_size < self.child_size:
            raise ValueError("父块大小不能小于子块大小。")

    @classmethod
    def from_env(cls) -> ChunkingConfig:
        defaults = asdict(cls())
        return cls(**{
            key: int(os.getenv(f"AGENT_RAG_{key.upper()}", str(value)))
            for key, value in defaults.items()
        })

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Keep paragraph boundaries while removing incidental indentation/CRLF."""
    normalized = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def split_ranges(
    text: str, size: int, overlap: int, *, paragraphs: bool = False,
) -> list[tuple[int, int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("分块 size/overlap 无效。")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if paragraphs and end < len(text):
            boundary = text.rfind("\n\n", start, end)
            # A tiny paragraph must not stall the overlap window.
            if boundary > start + max(overlap, size // 2):
                end = boundary + 2
        ranges.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return ranges


def slice_spans(spans: list[SourceSpan], start: int, end: int) -> list[SourceSpan]:
    return [
        SourceSpan(
            start=max(span.start, start) - start,
            end=min(span.end, end) - start,
            page_number=span.page_number,
        )
        for span in spans if max(span.start, start) < min(span.end, end)
    ]


def located_metadata(metadata: dict, spans: list[SourceSpan]) -> dict:
    result = dict(metadata)
    result.pop("page_number", None)
    result.pop("page_start", None)
    result.pop("page_end", None)
    pages = sorted({span.page_number for span in spans if span.page_number is not None})
    if pages:
        result.update(page_start=pages[0], page_end=pages[-1])
        if len(pages) == 1:
            result["page_number"] = pages[0]
    return result


def split_section(
    section: KnowledgeSection, config: ChunkingConfig,
) -> tuple[list[KnowledgeParent], list[KnowledgeChunk]]:
    # Loader/section text is normalized before page offsets are recorded.
    text = section.text
    base_id = section.chunk_id or f"{Path(section.source_path).stem}-s{section.section_index}"
    parent_ranges = split_ranges(text, config.parent_size, config.parent_overlap, paragraphs=True)
    parents: list[KnowledgeParent] = []
    children: list[KnowledgeChunk] = []
    for parent_index, (start, end) in enumerate(parent_ranges, 1):
        parent_id = f"{base_id}-parent-{parent_index}"
        spans = slice_spans(section.source_spans, start, end)
        metadata = located_metadata(section.metadata, spans)
        title = section.section
        if section.metadata.get("source_format") == "pdf":
            first, last = metadata["page_start"], metadata["page_end"]
            title = f"第 {first} 页" if first == last else f"第 {first}–{last} 页"
        metadata.update(section=title, parent_index=parent_index, parent_count=len(parent_ranges),
                        section_start=start, section_end=end)
        parent = KnowledgeParent(parent_id=parent_id, document_title=section.document_title,
                                 source_path=section.source_path, section=title,
                                 keywords=section.keywords, effective_status=section.effective_status,
                                 text=text[start:end], metadata=metadata, source_spans=spans)
        parents.append(parent)
        child_ranges = split_ranges(parent.text, config.child_size, config.child_overlap)
        for child_index, (child_start, child_end) in enumerate(child_ranges, 1):
            if len(parent_ranges) == 1:
                child_id = base_id if len(child_ranges) == 1 else f"{base_id}-c{child_index}"
            else:
                child_id = f"{parent_id}-child-{child_index}"
            child_spans = slice_spans(spans, child_start, child_end)
            children.append(KnowledgeChunk(
                chunk_id=child_id, parent_id=parent_id, parent_start=child_start, parent_end=child_end,
                document_title=parent.document_title, source_path=parent.source_path, section=parent.section,
                keywords=parent.keywords, effective_status=parent.effective_status,
                text=parent.text[child_start:child_end], source_spans=child_spans,
                metadata={**located_metadata(metadata, child_spans),
                          "chunk_index": child_index, "chunk_count": len(child_ranges)},
            ))
    return parents, children
