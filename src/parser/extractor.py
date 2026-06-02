from dataclasses import dataclass
from typing import List
from typing import List, Any, Dict
import fitz  # type: ignore


@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float

@dataclass
class ExtractedBlock:
    text: str
    font_size: float
    bbox: BBox
    page_number: int

class PDFExtractor:
    """
    Внутренний движок парсинга.
    Инкапсулирует сложную логику динамических колонок и очистки текста.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self) -> List[ExtractedBlock]:
        doc = fitz.open(self.pdf_path)
        extracted_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = page.get_text("dict")
            raw_blocks = page_dict.get("blocks", [])

            # Фильтруем только текстовые блоки (type == 0)
            text_blocks = [b for b in raw_blocks if b.get("type") == 0]
            if not text_blocks:
                continue

            # 1. Вычисляем динамическую сетку колонок по оси X
            x_intervals = [[b["bbox"][0], b["bbox"][2]] for b in text_blocks]
            x_intervals.sort(key=lambda x: x[0])

            columns_x = []
            if x_intervals:
                current_col = x_intervals[0]
                for i in range(1, len(x_intervals)):
                    interval = x_intervals[i]
                    # Допуск 5px для слипания отрезков в одну колонку
                    if interval[0] <= current_col[1] + 5:
                        current_col[1] = max(current_col[1], interval[1])
                    else:
                        columns_x.append(current_col)
                        current_col = interval
                columns_x.append(current_col)

            # 2. Распределяем текстовые блоки по найденным колонкам
            column_blocks: List[List[Dict[str, Any]]] = [[] for _ in range(len(columns_x))]
            for b in text_blocks:
                x0 = b["bbox"][0]
                assigned_col = 0
                for i, col in enumerate(columns_x):
                    if col[0] - 10 <= x0 <= col[1] + 10:
                        assigned_col = i
                        break
                column_blocks[assigned_col].append(b)

            # 3. Сортируем блоки внутри колонок строго сверху вниз (по Y)
            sorted_blocks = []
            for col_list in column_blocks:
                col_list.sort(key=lambda x: x["bbox"][1])
                sorted_blocks.extend(col_list)

            # 4. Формируем строгие объекты
            for b in sorted_blocks:
                x0, y0, x1, y1 = b["bbox"]
                bbox_obj = BBox(left=x0, top=y0, right=x1, bottom=y1)

                line_texts = []
                font_sizes = []

                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join([span["text"] for span in spans])
                    line_texts.append(line_text)
                    for span in line.get("spans", []):
                        font_sizes.append(span["size"])

                # Убиваем висячие переносы и склеиваем абзац
                clean_text = (
                    "\n".join(line_texts)
                    .replace("-\n", "")
                    .replace("\n", " ")
                    .strip()
                )

                if not clean_text:
                    continue

                extracted_blocks.append(
                    ExtractedBlock(
                        text=clean_text,
                        font_size=max(font_sizes) if font_sizes else 0.0,
                        bbox=bbox_obj,
                        page_number=page_num + 1
                    )
                )

        return extracted_blocks



def get_page_blocks(filepath: str) -> List[Dict[str, Any]]:
    """
    API-модуля. Абстрагирует команду от внутренней реализации.
    Принимает путь к PDF, возвращает стандартизированный список словарей.
    """
    extractor = PDFExtractor(filepath)
    blocks = extractor.extract()

    result = []
    for b in blocks:
        result.append({
            "text": b.text,
            "font_size": round(b.font_size, 1),
            "bbox": {
                "left": round(b.bbox.left, 2),
                "top": round(b.bbox.top, 2),
                "right": round(b.bbox.right, 2),
                "bottom": round(b.bbox.bottom, 2)
            },
            "page_number": b.page_number
        })

    return result
