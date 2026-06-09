"""Тесты логирования/сводки пайплайна (parser.pipeline)."""

from __future__ import annotations

from pathlib import Path

from parser.pipeline import (
    DocStatus,
    _missing_required,
    process_single_file,
    summarize,
)
from parser.schemas import Document, Metadata, Section


def _doc(**meta_over: object) -> Document:
    """Document с заполненными обязательными полями; поля можно переопределить."""
    base: dict[str, object] = {
        "title": "A Title",
        "title_en": None,
        "authors": ["Doe, Jane"],
        "abstract": "Some abstract.",
        "keywords": [],
        "doi": "10.1/x",
        "journal": None,
        "year": 2025,
        "metadata_source": "pdf",
        "metadata_confidence": 0.5,
    }
    base.update(meta_over)
    return Document(
        metadata=Metadata(**base),  # type: ignore[arg-type]
        sections=[
            Section(
                heading="Intro",
                level=1,
                content="text",
                subsections=[],
                number=None,
                status=None,
                status_effective_from=None,
            )
        ],
        figures=[],
        tables=[],
        equations=[],
        acknowledgments=None,
        raw_text="full text",
    )


def test_missing_required_all_present() -> None:
    assert _missing_required(_doc()) == []


def test_missing_required_reports_empty_fields() -> None:
    doc = _doc(authors=[], doi=None, abstract=None)
    missing = _missing_required(doc)
    assert set(missing) == {"authors", "doi", "abstract"}
    # порядок соответствует _REQUIRED_FIELDS (title, authors, abstract, doi, ...)
    assert missing == ["authors", "abstract", "doi"]


def test_missing_required_empty_sections_and_rawtext() -> None:
    doc = _doc()
    doc.sections = []
    doc.raw_text = "   "
    assert set(_missing_required(doc)) == {"sections", "raw_text"}


def test_summarize_counts_and_avg() -> None:
    results = [
        DocStatus(name="a", status="success", seconds=2.0),
        DocStatus(name="b", status="partial", seconds=4.0, missing=["doi"]),
        DocStatus(name="c", status="error", seconds=0.0, error="boom"),
        DocStatus(name="d", status="skipped"),
    ]
    summary = summarize(results, total_seconds=10.0)

    assert summary.total == 4
    assert summary.success == 1
    assert summary.partial == 1
    assert summary.errors == 1
    assert summary.skipped == 1
    assert summary.total_seconds == 10.0
    # среднее только по обработанным (a,b,c): (2+4+0)/3 = 2.0
    assert summary.avg_seconds == 2.0


def test_summarize_empty() -> None:
    summary = summarize([], total_seconds=0.0)
    assert summary.total == 0
    assert summary.avg_seconds == 0.0


def test_process_single_file_skips_existing(tmp_path: Path) -> None:
    # Если валидный JSON уже есть и нет --overwrite -> статус skipped (без обработки).
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "doc.json").write_text("{}", encoding="utf-8")

    status = process_single_file(pdf, tmp_path, overwrite=False, extract_images=False)

    assert status.status == "skipped"
    assert status.name == "doc.pdf"
