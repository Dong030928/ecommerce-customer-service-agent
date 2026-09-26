"""LangChain loaders for the supported repository-local knowledge formats."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".pdf", ".docx"})


def load_source_file(path: Path) -> list[Document]:
    """Extract text without deciding policy metadata or chunk boundaries."""

    # Import optional parser dependencies only for the format being loaded.
    if path.suffix.lower() in {".md", ".txt"}:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path), mode="page")
    elif path.suffix.lower() == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"不支持的知识文件格式：{path.name}")

    try:
        documents = loader.load()
    except Exception as exc:
        raise ValueError(f"知识文件解析失败：{path.name} ({type(exc).__name__})") from exc
    if not documents or any(not document.page_content.strip() for document in documents):
        raise ValueError(f"知识文件没有可提取的文本：{path.name}")
    return documents
