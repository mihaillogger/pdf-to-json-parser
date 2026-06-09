from typing import Any

import fitz
from loguru import logger

from parser.schemas import BBox, PageBlock


class PDFExtractor:
    """Извлекает текстовые и графические блоки из PDF-документа
    с учетом колонок и зонирования."""

    def __init__(
        self,
        pdf_path: str,
        col_tolerance: float = 2.0,
        block_tolerance: float = 10.0,
        spanning_threshold: float = 0.7,
    ):
        self.pdf_path = pdf_path
        self.col_tolerance = col_tolerance
        self.block_tolerance = block_tolerance
        self.spanning_threshold = spanning_threshold

    def extract(self) -> list[PageBlock]:
        extracted_blocks: list[PageBlock] = []

        try:
            with fitz.open(self.pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_width = page.rect.width
                    page_height = page.rect.height
                    raw_blocks = page.get_text("dict").get("blocks", [])

                    valid_blocks: list[dict[str, Any]] = []

                    for b in raw_blocks:
                        b_type = b.get("type")
                        if b_type not in (0, 1):
                            continue

                        x0, y0, x1, y1 = b["bbox"]

                        if b_type == 0:
                            if y0 < 50 or y1 > (page_height - 50):
                                continue

                        valid_blocks.append(b)

                    if not valid_blocks:
                        continue

                    valid_blocks.sort(key=lambda x: x["bbox"][1])

                    zones: list[list[dict[str, Any]]] = []
                    current_zone: list[dict[str, Any]] = []

                    for b in valid_blocks:
                        x0, y0, x1, y1 = b["bbox"]
                        is_spanning = (x1 - x0) > (page_width * self.spanning_threshold)

                        if is_spanning:
                            if current_zone:
                                zones.append(current_zone)
                                current_zone = []
                            zones.append([b])
                        else:
                            current_zone.append(b)

                    if current_zone:
                        zones.append(current_zone)

                    for zone in zones:
                        if len(zone) == 1:
                            z_bbox = zone[0]["bbox"]
                            is_wide = (z_bbox[2] - z_bbox[0]) > (
                                page_width * self.spanning_threshold
                            )
                            if is_wide:
                                ext = self._process_block(zone[0], page_num)
                                extracted_blocks.extend(ext)
                                continue

                        x_intervals = sorted(
                            [[b["bbox"][0], b["bbox"][2]] for b in zone],
                            key=lambda x: x[0],
                        )

                        columns_x = []
                        if x_intervals:
                            current_col = x_intervals[0]
                            for interval in x_intervals[1:]:
                                if interval[0] <= current_col[1] + self.col_tolerance:
                                    current_col[1] = max(current_col[1], interval[1])
                                else:
                                    columns_x.append(current_col)
                                    current_col = interval
                            columns_x.append(current_col)

                        column_blocks: list[list[dict[str, Any]]] = [
                            [] for _ in range(len(columns_x))
                        ]
                        for b in zone:
                            x0 = b["bbox"][0]
                            assigned_col = 0
                            for i, col in enumerate(columns_x):
                                left_b = col[0] - self.block_tolerance
                                right_b = col[1] + self.block_tolerance
                                if left_b <= x0 <= right_b:
                                    assigned_col = i
                                    break
                            column_blocks[assigned_col].append(b)

                        for col_list in column_blocks:
                            col_list.sort(key=lambda x: x["bbox"][1])
                            for b in col_list:
                                ext = self._process_block(b, page_num)
                                extracted_blocks.extend(ext)

        except Exception as e:
            logger.error(f"Ошибка парсинга {self.pdf_path}: {e}")

        return extracted_blocks

    def _process_block(self, b: dict[str, Any], page_num: int) -> list[PageBlock]:
        block_type = b.get("type")

        # Отделяем картинки, ставим is_bold=False по умолчанию
        if block_type == 1:
            x0, y0, x1, y1 = b["bbox"]
            return [
                PageBlock(
                    text=None,
                    font_size=None,
                    bbox=BBox(left=x0, top=y0, right=x1, bottom=y1),
                    page_number=page_num + 1,
                    block_type="image",
                    is_bold=False,
                )
            ]

        results: list[PageBlock] = []
        current_text: list[str] = []
        current_font: float | None = None
        current_bold: bool | None = None
        current_bbox: list[float] | None = None

        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = ""
            prev_x1: float | None = None
            span_fonts: list[float] = []

            for s in spans:
                if prev_x1 is not None and (s["bbox"][0] - prev_x1) > 4.0:
                    line_text += " "
                line_text += s["text"]
                prev_x1 = s["bbox"][2]
                span_fonts.append(s["size"])

            line_font = max(span_fonts) if span_fonts else 0.0

            line_bold = any(
                bool(s.get("flags", 0) & 16) or ("bold" in s.get("font", "").lower())
                for s in spans
            )
            line_bbox = list(line["bbox"])

            if current_font is None or current_bbox is None or current_bold is None:
                current_font = line_font
                current_bold = line_bold
                current_bbox = line_bbox
                current_text.append(line_text)
            elif abs(current_font - line_font) > 1.0 or current_bold != line_bold:
                new_block = self._create_text_pageblock(
                    current_text, current_font, current_bbox, page_num, current_bold
                )
                if new_block:
                    results.append(new_block)
                current_font = line_font
                current_bold = line_bold
                current_bbox = line_bbox
                current_text = [line_text]
            else:
                current_bbox[0] = min(current_bbox[0], line_bbox[0])
                current_bbox[1] = min(current_bbox[1], line_bbox[1])
                current_bbox[2] = max(current_bbox[2], line_bbox[2])
                current_bbox[3] = max(current_bbox[3], line_bbox[3])
                current_text.append(line_text)

        if (
            current_text
            and current_font is not None
            and current_bbox is not None
            and current_bold is not None
        ):
            new_block = self._create_text_pageblock(
                current_text, current_font, current_bbox, page_num, current_bold
            )
            if new_block:
                results.append(new_block)

        return results

    def _create_text_pageblock(
        self,
        text_lines: list[str],
        font: float,
        bbox: list[float],
        page_num: int,
        is_bold: bool,
    ) -> PageBlock | None:
        raw_text = "\n".join(text_lines).strip()

        if len(raw_text) <= 3:
            return None

        return PageBlock(
            text=raw_text,
            font_size=round(font, 1),
            bbox=BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3]),
            page_number=page_num + 1,
            block_type="text",
            is_bold=is_bold,
        )


def get_page_blocks(filepath: str) -> list[PageBlock]:
    """Извлекает блоки PDF; при отсутствии текстового слоя — OCR-фоллбэк."""
    blocks = PDFExtractor(filepath).extract()

    # Нет ни одного текстового блока — вероятно скан без текстового слоя.
    if not any(b.block_type == "text" and b.text for b in blocks):
        from parser.ocr import ocr_pdf

        ocr_blocks = ocr_pdf(filepath)
        if ocr_blocks:
            logger.info(f"Текстовый слой отсутствует, применён OCR: {filepath}")
            # Сохраняем картиночные блоки экстрактора + добавляем распознанный текст.
            blocks = blocks + ocr_blocks

    return blocks
