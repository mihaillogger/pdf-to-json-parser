"""Доп. тесты экстрактора: порядок чтения колонок, зоны, многостраничность,
публичный get_page_blocks + OCR-фоллбэк. Дополняют tests/test_extractor.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import parser.ocr
from parser.extractor import PDFExtractor, get_page_blocks
from parser.schemas import BBox, PageBlock


def _text_block(
    text: str, x0: float, y0: float, x1: float, y1: float
) -> dict[str, Any]:
    """Сырой текстовый блок fitz (type=0) с одним спаном."""
    return {
        "type": 0,
        "bbox": [x0, y0, x1, y1],
        "lines": [
            {
                "bbox": [x0, y0, x1, y1],
                "spans": [{"text": text, "size": 10.0, "bbox": [x0, y0, x1, y1]}],
            }
        ],
    }


def _make_page(blocks: list[dict[str, Any]], width: float, height: float) -> MagicMock:
    page = MagicMock()
    page.rect.width = width
    page.rect.height = height
    page.get_text.return_value = {"blocks": blocks}
    return page


def _patch_fitz(pages: list[MagicMock]) -> Any:
    """Контекст-патч fitz.open, отдающий документ из заданных страниц."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = len(pages)
    mock_doc.__getitem__.side_effect = lambda i: pages[i]
    cm = patch("fitz.open")
    mock_open = cm.start()
    mock_open.return_value.__enter__.return_value = mock_doc
    return cm


def test_two_column_reading_order() -> None:
    """Левая колонка читается целиком до правой (а не построчно через всю ширину)."""
    blocks = [
        _text_block("Left column first line", 50, 100, 250, 130),
        _text_block("Right column first line", 320, 100, 550, 130),
        _text_block("Left column second line", 50, 200, 250, 230),
        _text_block("Right column second line", 320, 200, 550, 230),
    ]
    cm = _patch_fitz([_make_page(blocks, 600.0, 800.0)])
    try:
        result = PDFExtractor("dummy.pdf").extract()
    finally:
        cm.stop()

    texts = [b.text for b in result]
    assert texts == [
        "Left column first line",
        "Left column second line",
        "Right column first line",
        "Right column second line",
    ]


def test_full_width_block_separates_columns() -> None:
    """Полноширинный заголовок идёт первым, затем левая и правая колонки."""
    blocks = [
        _text_block("Full Width Spanning Title Block", 50, 60, 560, 95),
        _text_block("Left A", 50, 120, 250, 150),
        _text_block("Right A", 320, 120, 550, 150),
        _text_block("Left B", 50, 220, 250, 250),
        _text_block("Right B", 320, 220, 550, 250),
    ]
    cm = _patch_fitz([_make_page(blocks, 600.0, 800.0)])
    try:
        result = PDFExtractor("dummy.pdf").extract()
    finally:
        cm.stop()

    texts = [b.text for b in result]
    assert texts[0] == "Full Width Spanning Title Block"
    assert texts == [
        "Full Width Spanning Title Block",
        "Left A",
        "Left B",
        "Right A",
        "Right B",
    ]


def test_multipage_page_numbers() -> None:
    """page_number соответствует странице (1-индексация) на нескольких страницах."""
    page1 = _make_page(
        [_text_block("Content on page one", 50, 100, 400, 130)], 600.0, 800.0
    )
    page2 = _make_page(
        [_text_block("Content on page two", 50, 100, 400, 130)], 600.0, 800.0
    )
    cm = _patch_fitz([page1, page2])
    try:
        result = PDFExtractor("dummy.pdf").extract()
    finally:
        cm.stop()

    assert len(result) == 2
    assert result[0].page_number == 1
    assert result[1].page_number == 2


def test_get_page_blocks_uses_extract_when_text_present() -> None:
    """Если текстовый слой есть — OCR-фоллбэк не вызывается."""
    page = _make_page(
        [_text_block("Real extracted text", 50, 100, 400, 130)], 600.0, 800.0
    )
    cm = _patch_fitz([page])
    try:
        with patch.object(parser.ocr, "ocr_pdf") as mock_ocr:
            result = get_page_blocks("dummy.pdf")
            mock_ocr.assert_not_called()
    finally:
        cm.stop()

    assert any(b.text == "Real extracted text" for b in result)


def test_get_page_blocks_falls_back_to_ocr_when_empty() -> None:
    """Нет текстового слоя -> используется OCR-фоллбэк."""
    ocr_block = PageBlock(
        text="OCR recovered text",
        font_size=12.0,
        bbox=BBox(left=10.0, top=10.0, right=200.0, bottom=30.0),
        page_number=1,
        block_type="text",
        is_bold=False,
    )
    # пустая страница -> extract() возвращает []
    cm = _patch_fitz([_make_page([], 600.0, 800.0)])
    try:
        with patch.object(parser.ocr, "ocr_pdf", return_value=[ocr_block]) as mock_ocr:
            result = get_page_blocks("dummy.pdf")
            mock_ocr.assert_called_once()
    finally:
        cm.stop()

    assert [b.text for b in result] == ["OCR recovered text"]
