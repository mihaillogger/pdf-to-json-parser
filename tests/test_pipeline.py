"""Тесты логирования/сводки пайплайна (parser.pipeline)."""

from __future__ import annotations

from pathlib import Path

from parser.pipeline import (
    DocStatus,
    _missing_required,
    enrich_visual_captions,
    process_single_file,
    summarize,
)
from parser.schemas import BBox, Document, Figure, Metadata, PageBlock, Section, Table


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


# --- Сопоставление подписей фигур/таблиц с текстовым слоем (enrich_visual_captions) ---


def _bbox(left: float, top: float, right: float, bottom: float) -> BBox:
    return BBox(left=left, top=top, right=right, bottom=bottom)


def _text_block(text: str, bbox: BBox, page: int = 1) -> PageBlock:
    return PageBlock(
        text=text,
        font_size=10.0,
        bbox=bbox,
        page_number=page,
        block_type="text",
        is_bold=False,
    )


def _figure(bbox: BBox, page: int = 1) -> Figure:
    return Figure(
        id="Figure 1", caption="", page=page, bbox=bbox, img_path="x.png", panels=[]
    )


def _table(bbox: BBox, page: int = 1) -> Table:
    return Table(
        id="Table 1", caption="", page=page, bbox=bbox, img_path="x.png", data=[]
    )


def test_caption_assigned_to_figure_below() -> None:
    # Подпись расположена прямо под фигурой и горизонтально перекрывается -> матч.
    fig = _figure(_bbox(50, 100, 250, 300))
    caption = _text_block(
        "Figure 2. Reaction scheme overview.", _bbox(50, 305, 250, 320)
    )
    body = _text_block("Body paragraph text here.", _bbox(50, 400, 250, 450))

    enrich_visual_captions([fig], [], [body, caption])

    assert fig.id == "Figure 2"
    assert fig.caption == "Figure 2. Reaction scheme overview."


def test_caption_supplementary_id_preserved() -> None:
    # Номера с префиксом S (supplementary) сохраняются: "Fig. S1" -> id "Figure S1".
    fig = _figure(_bbox(50, 100, 250, 300))
    caption = _text_block(
        "Fig. S1. Device fabrication steps.", _bbox(50, 305, 250, 320)
    )

    enrich_visual_captions([fig], [], [caption])

    assert fig.id == "Figure S1"


def test_caption_assigned_to_table_above() -> None:
    # Подпись таблицы обычно сверху; берём из «табличного» пула.
    tab = _table(_bbox(50, 200, 250, 400))
    caption = _text_block("Table 3. Measured yields.", _bbox(50, 180, 250, 195))

    enrich_visual_captions([], [tab], [caption])

    assert tab.id == "Table 3"
    assert tab.caption == "Table 3. Measured yields."


def test_figure_and_table_do_not_steal_each_other() -> None:
    # Фигура не должна получить подпись таблицы и наоборот.
    fig = _figure(_bbox(50, 100, 250, 300))
    tab = _table(_bbox(50, 400, 250, 600))
    fig_cap = _text_block("Figure 1. Spectra.", _bbox(50, 305, 250, 320))
    tab_cap = _text_block("Table 1. Parameters.", _bbox(50, 380, 250, 395))

    enrich_visual_captions([fig], [tab], [fig_cap, tab_cap])

    assert fig.caption == "Figure 1. Spectra."
    assert tab.caption == "Table 1. Parameters."


def test_distant_caption_not_assigned() -> None:
    # Слишком далеко по вертикали (> порога) — подпись не привязывается.
    fig = _figure(_bbox(50, 100, 250, 150))
    far_caption = _text_block("Figure 9. Unrelated.", _bbox(50, 700, 250, 720))

    enrich_visual_captions([fig], [], [far_caption])

    assert fig.id == "Figure 1"  # дефолт не тронут
    assert fig.caption == ""


def test_two_figures_get_distinct_captions() -> None:
    # Две фигуры на странице -> каждая берёт свою ближайшую подпись (без дублей).
    fig_top = _figure(_bbox(50, 100, 250, 200))
    fig_bottom = _figure(_bbox(50, 400, 250, 500))
    cap_top = _text_block("Figure 1. Top.", _bbox(50, 205, 250, 220))
    cap_bottom = _text_block("Figure 2. Bottom.", _bbox(50, 505, 250, 520))

    enrich_visual_captions([fig_top, fig_bottom], [], [cap_bottom, cap_top])

    assert fig_top.caption == "Figure 1. Top."
    assert fig_bottom.caption == "Figure 2. Bottom."
