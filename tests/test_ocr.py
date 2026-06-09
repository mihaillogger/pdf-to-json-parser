"""Тесты OCR-фоллбэка (parser.ocr) — без зависимости от бинаря Tesseract."""

from __future__ import annotations

from typing import Any

from parser.ocr import _group_words_into_blocks, ocr_pdf


def _tess_data(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Собирает структуру в формате pytesseract.image_to_data(DICT)."""
    keys = [
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ]
    return {k: [r[k] for r in rows] for k in keys}


def test_group_words_into_blocks_joins_line() -> None:
    # Два слова одной строки -> один блок с объединённым текстом.
    data = _tess_data(
        [
            {"block_num": 1, "par_num": 1, "line_num": 1, "left": 100, "top": 200,
             "width": 40, "height": 20, "conf": 95, "text": "Hello"},
            {"block_num": 1, "par_num": 1, "line_num": 1, "left": 150, "top": 200,
             "width": 50, "height": 20, "conf": 90, "text": "World"},
        ]
    )
    blocks = _group_words_into_blocks(data, page_number=1, scale=72.0 / 144.0)

    assert len(blocks) == 1
    assert blocks[0].text == "Hello World"
    assert blocks[0].page_number == 1
    assert blocks[0].block_type == "text"
    # scale=0.5: left 100px -> 50pt, height 20px -> font_size 10pt
    assert blocks[0].bbox.left == 50.0
    assert blocks[0].font_size == 10.0


def test_group_words_skips_low_conf_and_empty() -> None:
    data = _tess_data(
        [
            {"block_num": 1, "par_num": 1, "line_num": 1, "left": 0, "top": 0,
             "width": 10, "height": 10, "conf": -1, "text": "ghost"},
            {"block_num": 1, "par_num": 1, "line_num": 1, "left": 0, "top": 0,
             "width": 10, "height": 10, "conf": 50, "text": "   "},
        ]
    )
    assert _group_words_into_blocks(data, page_number=1, scale=1.0) == []


def test_group_words_separate_lines() -> None:
    data = _tess_data(
        [
            {"block_num": 1, "par_num": 1, "line_num": 1, "left": 10, "top": 300,
             "width": 60, "height": 20, "conf": 90, "text": "Second"},
            {"block_num": 1, "par_num": 1, "line_num": 2, "left": 10, "top": 100,
             "width": 60, "height": 20, "conf": 90, "text": "First"},
        ]
    )
    blocks = _group_words_into_blocks(data, page_number=2, scale=1.0)
    # Две разные строки, отсортированы сверху вниз (по top).
    assert [b.text for b in blocks] == ["First", "Second"]


def test_ocr_pdf_missing_file_returns_empty() -> None:
    # Нет файла/Tesseract -> мягкая деградация, пустой список (без исключения).
    assert ocr_pdf("/nonexistent/file.pdf") == []
