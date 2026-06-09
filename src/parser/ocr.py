"""OCR-фоллбэк для документов без текстового слоя.

Если в PDF нет извлекаемого текста (скан-изображения), страницы рендерятся в
картинки и распознаются Tesseract в двуязычном режиме (eng+rus). Результат —
те же :class:`parser.schemas.PageBlock`, что и у обычного экстрактора, поэтому
остальной пайплайн (метаданные, секции) работает без изменений.

Деградирует мягко: если Tesseract/Pillow недоступны или распознавание упало —
возвращается пустой список и пишется предупреждение (пайплайн не падает).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import fitz
from loguru import logger

from parser.schemas import BBox, PageBlock

#: Язык(и) распознавания: английский + русский (в корпусе есть кириллица).
OCR_LANG = "eng+rus"

#: Разрешение рендера страницы в пиксели на дюйм.
OCR_DPI = 200

#: Минимальная уверенность слова Tesseract (conf), ниже — отбрасываем.
_MIN_WORD_CONF = 0.0


def _group_words_into_blocks(
    data: dict[str, list[Any]], page_number: int, scale: float
) -> list[PageBlock]:
    """Группирует слова Tesseract (image_to_data) в построчные ``PageBlock``.

    Args:
        data: Результат ``pytesseract.image_to_data(..., output_type=DICT)``.
        page_number: Номер страницы (1-индексация).
        scale: Коэффициент пиксели->точки PDF (``72 / dpi``).

    Returns:
        Список текстовых блоков, отсортированных сверху вниз.
    """
    lines: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < _MIN_WORD_CONF:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines[key].append(i)

    blocks: list[PageBlock] = []
    for idxs in lines.values():
        idxs.sort(key=lambda i: data["left"][i])
        text = " ".join(str(data["text"][i]).strip() for i in idxs).strip()
        if len(text) <= 1:
            continue
        left = min(data["left"][i] for i in idxs)
        top = min(data["top"][i] for i in idxs)
        right = max(data["left"][i] + data["width"][i] for i in idxs)
        bottom = max(data["top"][i] + data["height"][i] for i in idxs)
        height_px = bottom - top
        blocks.append(
            PageBlock(
                text=text,
                # Высота строки в точках — прокси «размера шрифта» для эвристик.
                font_size=round(height_px * scale, 1) if height_px else None,
                bbox=BBox(
                    left=left * scale,
                    top=top * scale,
                    right=right * scale,
                    bottom=bottom * scale,
                ),
                page_number=page_number,
                block_type="text",
                is_bold=False,
            )
        )
    blocks.sort(key=lambda b: b.bbox.top)
    return blocks


def page_needs_ocr(page: Any) -> bool:
    """Страница без извлекаемого текстового слоя (скан) -> нужен OCR."""
    return not page.get_text().strip()


def ocr_pdf(
    filepath: str,
    *,
    lang: str = OCR_LANG,
    dpi: int = OCR_DPI,
    only_pages_without_text: bool = True,
) -> list[PageBlock]:
    """Распознаёт текст PDF через Tesseract и отдаёт ``PageBlock``-и.

    Args:
        filepath: Путь к PDF.
        lang: Языки Tesseract (по умолчанию ``eng+rus``).
        dpi: Разрешение рендера страниц.
        only_pages_without_text: OCR только страниц без текстового слоя.

    Returns:
        Список распознанных блоков. Пустой список при недоступности OCR или ошибке.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError:
        logger.warning("OCR пропущен: не установлены pytesseract/Pillow")
        return []

    scale = 72.0 / dpi
    result: list[PageBlock] = []
    try:
        with fitz.open(filepath) as doc:
            for page_index, page in enumerate(doc):
                if only_pages_without_text and not page_needs_ocr(page):
                    continue
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
                data = pytesseract.image_to_data(
                    image, lang=lang, output_type=pytesseract.Output.DICT
                )
                result.extend(_group_words_into_blocks(data, page_index + 1, scale))
    except Exception as exc:
        logger.warning(f"OCR-фоллбэк не сработал для {filepath}: {exc}")
        return []

    if result:
        logger.info(f"OCR распознал {len(result)} блоков в {filepath}")
    return result
