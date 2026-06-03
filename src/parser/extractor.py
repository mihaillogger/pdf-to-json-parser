import logging
from typing import Any, List

import fitz  # type: ignore

# Импортируем строгие Pydantic-схемы, которые требует Арс
from parser.schemas import BBox, PageBlock

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Внутренний движок парсинга PDF-документов.
    Анализирует геометрию страницы, определяет колонки
    и выстраивает текст в правильном порядке чтения.
    """

    def __init__(
        self,
        pdf_path: str,
        col_tolerance: float = 5.0,
        block_tolerance: float = 10.0,
    ):
        self.pdf_path = pdf_path
        self.col_tolerance = col_tolerance
        self.block_tolerance = block_tolerance

    def extract(self) -> List[PageBlock]:
        extracted_blocks = []

        try:
            with fitz.open(self.pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    raw_blocks = page.get_text("dict").get("blocks", [])

                    text_blocks = [b for b in raw_blocks if b.get("type") == 0]
                    if not text_blocks:
                        continue

                    # 1. Вычисляем динамическую сетку колонок по оси X
                    x_intervals = sorted(
                        [[b["bbox"][0], b["bbox"][2]] for b in text_blocks],
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

                    # 2. Распределяем текстовые блоки по найденным колонкам
                    column_blocks: List[List[dict[str, Any]]] = [
                        [] for _ in range(len(columns_x))
                    ]
                    for b in text_blocks:
                        x0 = b["bbox"][0]
                        assigned_col = 0
                        for i, col in enumerate(columns_x):
                            left_b = col[0] - self.block_tolerance
                            right_b = col[1] + self.block_tolerance
                            if left_b <= x0 <= right_b:
                                assigned_col = i
                                break
                        column_blocks[assigned_col].append(b)

                    # 3. Сортируем блоки внутри колонок строго сверху вниз
                    for col_list in column_blocks:
                        col_list.sort(key=lambda x: x["bbox"][1])

                        for b in col_list:
                            x0, y0, x1, y1 = b["bbox"]

                            line_texts = []
                            font_sizes = []

                            for line in b.get("lines", []):
                                spans = line.get("spans", [])
                                text_span = "".join([s["text"] for s in spans])
                                line_texts.append(text_span)
                                font_sizes.extend([s["size"] for s in spans])

                            # Склеиваем строки просто через \n.
                            # Pydantic-валидатор сам удалит висячие дефисы!
                            raw_text = "\n".join(line_texts)

                            if raw_text.strip():
                                # Упаковываем в схему (Pydantic проверит типы)
                                extracted_blocks.append(
                                    PageBlock(
                                        text=raw_text,
                                        font_size=(
                                            max(font_sizes) if font_sizes else 0.0
                                        ),
                                        bbox=BBox(
                                            left=x0, top=y0, right=x1, bottom=y1
                                        ),
                                        page_number=page_num + 1,
                                    )
                                )

        except Exception as e:
            logger.error(
                f"Произошла ошибка при обработке файла {self.pdf_path}: {e}"
            )

        return extracted_blocks


def get_page_blocks(filepath: str) -> List[PageBlock]:
    """
    API-модуля.
    Возвращает стандартизированный список Pydantic-объектов.
    """
    extractor = PDFExtractor(filepath)
    return extractor.extract()
