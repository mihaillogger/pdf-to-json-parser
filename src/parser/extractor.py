import logging
from typing import Any, Dict, List, Optional

import fitz  # type: ignore

from parser.schemas import BBox, PageBlock

logger = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(
        self,
        pdf_path: str,
        col_tolerance: float = 5.0,
        block_tolerance: float = 10.0,
        spanning_threshold: float = 0.7,
    ):
        self.pdf_path = pdf_path
        self.col_tolerance = col_tolerance
        self.block_tolerance = block_tolerance
        self.spanning_threshold = spanning_threshold

    def extract(self) -> List[PageBlock]:
        extracted_blocks: List[PageBlock] = []

        try:
            with fitz.open(self.pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_width = page.rect.width
                    raw_blocks = page.get_text("dict").get("blocks", [])

                    valid_blocks = [
                        b for b in raw_blocks if b.get("type") in (0, 1)
                    ]
                    if not valid_blocks:
                        continue

                    valid_blocks.sort(key=lambda x: x["bbox"][1])

                    # --- ЭТАП 1: ГОРИЗОНТАЛЬНОЕ ЗОНИРОВАНИЕ ---
                    zones: List[List[Dict[str, Any]]] = []
                    current_zone: List[Dict[str, Any]] = []

                    for b in valid_blocks:
                        x0, y0, x1, y1 = b["bbox"]
                        is_spanning = (x1 - x0) > (
                            page_width * self.spanning_threshold
                        )

                        if is_spanning:
                            if current_zone:
                                zones.append(current_zone)
                                current_zone = []
                            # Сквозной блок живет в своей отдельной зоне
                            zones.append([b])
                        else:
                            current_zone.append(b)

                    if current_zone:
                        zones.append(current_zone)

                    # --- ЭТАП 2: РАЗБОР КОЛОНОК ВНУТРИ ЗОН ---
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
                                if (
                                    interval[0]
                                    <= current_col[1] + self.col_tolerance
                                ):
                                    current_col[1] = max(
                                        current_col[1], interval[1]
                                    )
                                else:
                                    columns_x.append(current_col)
                                    current_col = interval
                            columns_x.append(current_col)

                        column_blocks: List[List[Dict[str, Any]]] = [
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

    def _process_block(
        self, b: Dict[str, Any], page_num: int
    ) -> List[PageBlock]:
        """
        Умная обработка одного блока: отделяет картинки
        и дробит текст при смене шрифта.
        """
        block_type = b.get("type")

        if block_type == 1:
            x0, y0, x1, y1 = b["bbox"]
            return [
                PageBlock(
                    text=None,
                    font_size=None,
                    bbox=BBox(left=x0, top=y0, right=x1, bottom=y1),
                    page_number=page_num + 1,
                    block_type="image",
                )
            ]

        results: List[PageBlock] = []
        current_text: List[str] = []
        current_font: Optional[float] = None
        current_bbox: Optional[List[float]] = None

        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join([s["text"] for s in spans])
            line_font = max([s["size"] for s in spans])
            line_bbox = list(line["bbox"])

            # Явная проверка на None гарантирует Mypy, что переменные инициализированы
            if current_font is None or current_bbox is None:
                current_font = line_font
                current_bbox = line_bbox
                current_text.append(line_text)
            elif abs(current_font - line_font) > 1.0:
                new_block = self._create_text_pageblock(
                    current_text, current_font, current_bbox, page_num
                )
                results.append(new_block)
                current_font = line_font
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
        ):
            new_block = self._create_text_pageblock(
                current_text, current_font, current_bbox, page_num
            )
            results.append(new_block)

        return results

    def _create_text_pageblock(
        self,
        text_lines: List[str],
        font: float,
        bbox: List[float],
        page_num: int,
    ) -> PageBlock:
        raw_text = "\n".join(text_lines).strip()
        return PageBlock(
            text=raw_text,
            font_size=round(font, 1),
            bbox=BBox(
                left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3]
            ),
            page_number=page_num + 1,
            block_type="text",
        )


def get_page_blocks(filepath: str) -> List[PageBlock]:
    return PDFExtractor(filepath).extract()
