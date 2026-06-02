import fitz  # type: ignore
import logging
from dataclasses import dataclass
from typing import List, Any, Dict

# Настройка логгера для отслеживания процесса парсинга и фиксации ошибок
logger = logging.getLogger(__name__)

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
    Внутренний движок парсинга PDF-документов.
    Анализирует геометрию страницы, определяет колонки и выстраивает текст в правильном порядке чтения.
    """
    def __init__(self, pdf_path: str, col_tolerance: float = 5.0, block_tolerance: float = 10.0):
        self.pdf_path = pdf_path
        # col_tolerance: Максимальное расстояние по горизонтали для объединения блоков в одну колонку
        self.col_tolerance = col_tolerance
        # block_tolerance: Погрешность при проверке принадлежности блока к найденной колонке
        self.block_tolerance = block_tolerance

    def extract(self) -> List[ExtractedBlock]:
        extracted_blocks = []

        try:
            # Использование контекстного менеджера гарантирует закрытие файла и предотвращает утечку ресурсов
            with fitz.open(self.pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    raw_blocks = page.get_text("dict").get("blocks", [])

                    # Оставляем только текстовые элементы, игнорируя изображения и графику (type == 0)
                    text_blocks = [b for b in raw_blocks if b.get("type") == 0]
                    if not text_blocks:
                        continue

                    # Этап 1: Определение границ колонок по оси X
                    # Сортируем блоки слева направо для последовательного формирования сетки
                    x_intervals = sorted([[b["bbox"][0], b["bbox"][2]] for b in text_blocks], key=lambda x: x[0])

                    columns_x = []
                    if x_intervals:
                        current_col = x_intervals[0]
                        for interval in x_intervals[1:]:
                            # Если текущий блок находится в пределах допуска, расширяем границы колонки
                            if interval[0] <= current_col[1] + self.col_tolerance:
                                current_col[1] = max(current_col[1], interval[1])
                            else:
                                # Если расстояние превышает допуск, фиксируем текущую колонку и начинаем новую
                                columns_x.append(current_col)
                                current_col = interval
                        columns_x.append(current_col)

                    # Этап 2: Распределение текстовых блоков по сформированным колонкам
                    column_blocks: List[List[Dict[str, Any]]] = [[] for _ in range(len(columns_x))]
                    for b in text_blocks:
                        x0 = b["bbox"][0]
                        assigned_col = 0
                        # Проверяем вхождение координаты X блока в границы каждой известной колонки
                        for i, col in enumerate(columns_x):
                            if col[0] - self.block_tolerance <= x0 <= col[1] + self.block_tolerance:
                                assigned_col = i
                                break
                        column_blocks[assigned_col].append(b)

                    # Этап 3: Сортировка блоков внутри колонок и формирование итоговой структуры
                    for col_list in column_blocks:
                        # Сортируем блоки внутри колонки сверху вниз (по оси Y) для соблюдения логики чтения
                        col_list.sort(key=lambda x: x["bbox"][1])
                        
                        for b in col_list:
                            x0, y0, x1, y1 = b["bbox"]
                            
                            line_texts = []
                            font_sizes = []

                            # Извлекаем текст и размеры шрифтов из каждой строки блока
                            for line in b.get("lines", []):
                                spans = line.get("spans", [])
                                line_texts.append("".join([span["text"] for span in spans]))
                                font_sizes.extend([span["size"] for span in spans])

                            # Очистка текста: удаление висячих переносов и объединение строк в единый абзац
                            clean_text = (
                                "\n".join(line_texts)
                                .replace("-\n", "")
                                .replace("\n", " ")
                                .strip()
                            )

                            # Игнорируем пустые блоки, не несущие смысловой нагрузки
                            if clean_text:
                                extracted_blocks.append(
                                    ExtractedBlock(
                                        text=clean_text,
                                        font_size=max(font_sizes) if font_sizes else 0.0,
                                        bbox=BBox(left=x0, top=y0, right=x1, bottom=y1),
                                        page_number=page_num + 1
                                    )
                                )
                                
        except Exception as e:
            logger.error(f"Произошла ошибка при обработке файла {self.pdf_path}: {e}")
            # Возвращаем частично собранные данные в случае критического сбоя библиотеки
            
        return extracted_blocks


def get_page_blocks(filepath: str) -> List[Dict[str, Any]]:
    """
    Публичный API модуля. 
    Изолирует вызывающий код от деталей реализации парсинга и геометрии документа.
    Возвращает стандартизированный список словарей, готовый к дальнейшему анализу.
    """
    extractor = PDFExtractor(filepath)
    
    # Конвертация внутренних объектов данных в требуемый JSON-совместимый формат
    return [
        {
            "text": b.text,
            "font_size": round(b.font_size, 1),
            "bbox": {
                "left": round(b.bbox.left, 2),
                "top": round(b.bbox.top, 2),
                "right": round(b.bbox.right, 2),
                "bottom": round(b.bbox.bottom, 2)
            },
            "page_number": b.page_number
        }
        for b in extractor.extract()
    ]