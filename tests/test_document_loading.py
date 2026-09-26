"""Real-format ingestion tests for the LangChain document-loader boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from rag import knowledge_base  # noqa: E402
from rag.hybrid_retrieval import retrieve_keyword_candidates  # noqa: E402
from rag.index_cache import (  # noqa: E402
    build_knowledge_index,
    get_knowledge_index,
    rebuild_knowledge_index,
    reset_index_and_cache,
)
from rag.prompting import build_citations  # noqa: E402
from rag.planning import build_retrieval_plan  # noqa: E402
from rag.query_rewrite import rewrite_retrieval_query  # noqa: E402
from api.schemas import KnowledgeHit  # noqa: E402


def write_manifest(path: Path, *, status: str = "active", document_id: str = "sample-policy") -> None:
    path.with_name(f"{path.name}.meta.yaml").write_text(
        "\n".join(
            [
                f"document_id: {document_id}",
                "title: 示例政策",
                "domain: shipping",
                f"effective_status: {status}",
                "owner: logistics_team",
                "tags: [物流, 时效]",
            ]
        ),
        encoding="utf-8",
    )


def write_text_pdf(path: Path, text: str) -> None:
    """Write a small text-layer PDF without adding a test-only dependency."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    with path.open("wb") as output:
        writer.write(output)


def write_text_docx(path: Path, text: str) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )


class DocumentLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_index_and_cache()
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.patch = patch.object(knowledge_base, "KNOWLEDGE_DIR", self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(reset_index_and_cache)

    def test_markdown_frontmatter_and_explicit_chunk_id_are_preserved(self) -> None:
        (self.directory / "policy.md").write_text(
            "---\ntitle: 示例政策\ndomain: shipping\neffective_status: active\n"
            "owner: logistics_team\ntags: [物流]\n---\n# 示例政策\n"
            "## 发货时效\n<!-- chunk_id: stable-shipping; keywords: 发货,时效 -->\n"
            "现货支付后安排发货。\n",
            encoding="utf-8",
        )

        chunks = knowledge_base.load_knowledge_chunks()

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["stable-shipping"])
        self.assertEqual(chunks[0].metadata["domain"], "shipping")
        self.assertEqual(chunks[0].section, "发货时效")

    def test_txt_loader_preserves_policy_metadata_and_citation(self) -> None:
        path = self.directory / "shipping.txt"
        path.write_text("配送时效以物流系统为准。", encoding="utf-8")
        write_manifest(path)

        chunks = knowledge_base.load_knowledge_chunks()
        citation = build_citations([KnowledgeHit(chunk=chunks[0], score=0.8)])[0]

        self.assertEqual(chunks[0].chunk_id, "sample-policy-s1")
        self.assertEqual(chunks[0].keywords, ["物流", "时效"])
        self.assertEqual(citation.source_path, "shipping.txt")
        self.assertEqual(citation.section, "示例政策")

        index = build_knowledge_index(chunks)
        rewrite = rewrite_retrieval_query("配送物流时效", "unknown")
        plan = build_retrieval_plan(rewrite, "unknown")
        hits = retrieve_keyword_candidates(plan, index=index)
        self.assertEqual(hits[0].chunk.chunk_id, "sample-policy-s1")

    def test_pdf_loader_keeps_one_based_page_location(self) -> None:
        path = self.directory / "shipping.pdf"
        write_text_pdf(path, "Shipping takes two days.")
        write_manifest(path)

        chunks = knowledge_base.load_knowledge_chunks()

        self.assertEqual(chunks[0].chunk_id, "sample-policy-p1")
        self.assertEqual(chunks[0].section, "第 1 页")
        self.assertEqual(chunks[0].metadata["page_number"], 1)
        self.assertIn("Shipping takes two days", chunks[0].text)

    def test_docx_loader_extracts_plain_text(self) -> None:
        path = self.directory / "shipping.docx"
        write_text_docx(path, "Shipping after payment")
        write_manifest(path)

        chunks = knowledge_base.load_knowledge_chunks()

        self.assertEqual(chunks[0].chunk_id, "sample-policy-s1")
        self.assertIn("Shipping after payment", chunks[0].text)

    def test_missing_or_invalid_manifest_fails_closed(self) -> None:
        path = self.directory / "shipping.txt"
        path.write_text("当前政策。", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺少元数据清单"):
            knowledge_base.load_knowledge_chunks()

        write_manifest(path, status="unknown")
        with self.assertRaisesRegex(ValueError, "effective_status 无效"):
            knowledge_base.load_knowledge_chunks()

        write_manifest(path)
        with path.with_name(f"{path.name}.meta.yaml").open("a", encoding="utf-8") as manifest:
            manifest.write("\npage_number: 99\n")
        with self.assertRaisesRegex(ValueError, "不支持的字段"):
            knowledge_base.load_knowledge_chunks()

    def test_textless_pdf_fails_instead_of_partial_index(self) -> None:
        path = self.directory / "scanned.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as output:
            writer.write(output)
        write_manifest(path)

        with self.assertRaisesRegex(ValueError, "没有可提取的文本"):
            knowledge_base.load_knowledge_chunks()

    def test_malformed_docx_reports_source_file(self) -> None:
        path = self.directory / "broken.docx"
        path.write_text("not a Word document", encoding="utf-8")
        write_manifest(path)

        with self.assertRaisesRegex(ValueError, "知识文件解析失败：broken.docx"):
            knowledge_base.load_knowledge_chunks()

    def test_duplicate_imported_chunk_id_is_rejected(self) -> None:
        for name in ("first.txt", "second.txt"):
            path = self.directory / name
            path.write_text(name, encoding="utf-8")
            write_manifest(path)

        with self.assertRaisesRegex(ValueError, "重复 chunk_id"):
            build_knowledge_index(knowledge_base.load_knowledge_chunks())

    def test_manifest_change_versions_index_and_failed_rebuild_keeps_old_snapshot(self) -> None:
        path = self.directory / "shipping.txt"
        path.write_text("配送时效以物流系统为准。", encoding="utf-8")
        write_manifest(path)
        original = get_knowledge_index()

        write_manifest(path, status="expired")
        updated = rebuild_knowledge_index()
        self.assertNotEqual(updated.version, original.version)
        self.assertEqual(updated.chunks_by_id["sample-policy-s1"].effective_status, "expired")

        path.with_name(f"{path.name}.meta.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "缺少元数据清单"):
            rebuild_knowledge_index()
        self.assertEqual(get_knowledge_index().version, updated.version)
