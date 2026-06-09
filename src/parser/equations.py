"""
Модуль для извлечения уравнений из PDF-документов.
Использует кастомную модель YOLOv8 для детекции bounding boxes
и Pix2Tex (LaTeX OCR) для генерации LaTeX-кода.
"""

import io
import logging
import os
import re
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO

try:
    from pix2tex.cli import LatexOCR

    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False

# Строгий импорт Pydantic-схемы (согласно архитектуре команды)
from parser.schemas import BBox, Equation

logger = logging.getLogger(__name__)


class EquationExtractor:
    """
    Класс для парсинга математических формул из научных статей.

    Attributes:
        model (YOLO): Обученная модель YOLOv8 для поиска формул.
        math_ocr (LatexOCR | None): Модель Pix2Tex для распознавания текста формул.
    """

    def __init__(self, model_path: str) -> None:
        """
        Инициализирует парсер уравнений.

        Args:
            model_path (str): Абсолютный путь к весам обученной YOLOv8.

        Raises:
            FileNotFoundError: Если файл с весами модели не найден.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл весов YOLO не найден: {model_path}")

        logger.info("Загрузка кастомных весов YOLO: %s", model_path)
        self.model = YOLO(model_path)

        if PIX2TEX_AVAILABLE:
            logger.info("Загрузка локальной модели Pix2Tex (LaTeX OCR)...")
            self.math_ocr = LatexOCR()
        else:
            logger.warning("Pix2Tex не установлен. Поле latex будет пустым.")
            self.math_ocr = None

    def _clean_latex(self, raw_latex: str) -> str:
        """
        Очищает сырой LaTeX от лишних пробелов, сохраняя метки (N) или \\tag{N}.

        Args:
            raw_latex (str): Сырая строка от Pix2Tex.

        Returns:
            str: Очищенная LaTeX строка.
        """
        # Убираем только дикий визуальный мусор перед номерами, сами номера не трогаем
        cleaned = re.sub(
            r"(?:~|\\quad|\\qquad|\\ |\\!|\\,)+(\(\d+\)|\\tag\{\d+\})$",
            r" \1",
            raw_latex,
        )
        return cleaned.strip()

    def _extract_id(self, latex_str: str) -> Optional[str]:
        """
        Извлекает идентификатор уравнения. Поддерживает форматы (1) и \\tag{1}.

        Args:
            latex_str (str): Очищенная LaTeX строка.

        Returns:
            Optional[str]: Найденный ID в формате "(N)" или None.
        """
        # Ищем либо (число), либо \tag{число} в конце строки
        match = re.search(r"(?:\((\d+)\)|\\tag\{(\d+)\})$", latex_str.strip())
        if match:
            # match.group(1) сработает для (1), match.group(2) для \tag{1}
            num = match.group(1) or match.group(2)
            return f"({num})"
        return None

    def process_pdf(self, pdf_path: str) -> List[Equation]:
        """
        Обрабатывает PDF-документ, вырезает формулы и возвращает массив схем Equation.

        Args:
            pdf_path (str): Путь к целевому PDF-документу.

        Returns:
            List[Equation]: Массив провалидированных Pydantic-объектов.
        """
        extracted_equations: List[Equation] = []
        zoom = 3.0
        mat = fitz.Matrix(zoom, zoom)

        logger.info("Старт нейро-парсинга уравнений: %s", pdf_path)

        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]

                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_img = Image.open(io.BytesIO(img_data))

                yolo_res = self.model.predict(
                    page_img, conf=0.15, iou=0.4, verbose=False
                )
                boxes = yolo_res[0].boxes

                if len(boxes) == 0:
                    continue

                sorted_boxes = sorted(boxes.xyxy.tolist(), key=lambda b: b[1])

                for box in sorted_boxes:
                    x1, y1, x2, y2 = box
                    crop_box = (
                        max(0, x1 - 5),
                        max(0, y1 - 5),
                        min(page_img.width, x2 + 5),
                        min(page_img.height, y2 + 5),
                    )

                    if crop_box[2] - crop_box[0] < 20 or crop_box[3] - crop_box[1] < 10:
                        continue

                    eq_img = page_img.crop(crop_box)

                    bbox_obj = BBox(
                        left=round(crop_box[0] / zoom, 2),
                        top=round(crop_box[1] / zoom, 2),
                        right=round(crop_box[2] / zoom, 2),
                        bottom=round(crop_box[3] / zoom, 2),
                    )

                    latex_code = ""
                    if self.math_ocr:
                        try:
                            raw_latex = self.math_ocr(eq_img)
                            latex_code = self._clean_latex(raw_latex)
                        except Exception as e:
                            logger.error(
                                "Ошибка Pix2Tex на странице %d: %s", page_num + 1, e
                            )

                    if not latex_code and PIX2TEX_AVAILABLE:
                        continue

                    eq_id = self._extract_id(latex_code)

                    equation_obj = Equation(
                        id=eq_id,
                        latex=latex_code,
                        context=None,
                        page=page_num + 1,
                        bbox=bbox_obj,
                    )

                    extracted_equations.append(equation_obj)

        return extracted_equations
