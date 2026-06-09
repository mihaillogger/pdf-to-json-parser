"""Модуль извлечения визуальных элементов (Таблиц и Фигур) из PDF-документов.

Архитектура пайплайна:
1. DocLayout-YOLOv10 — Пространственная детекция bounding boxes (BBox).
2. PyMuPDF (fitz) — Прецизионное кадрирование элементов (300 DPI для таблиц).
3. LLaVA (через Ollama) — Мультимодальный анализ и OCR-извлечение табличных данных.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import fitz
import ollama
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from pydantic import ValidationError

from parser.schemas import BBox, Figure, Panel, Table, TableDataResponse

logger = logging.getLogger(__name__)


class VLMTableExtractor:
    """Извлекатель табличных данных на базе локальной VLM (LLaVA).

    Выполняет конвертацию графических кропов таблиц в структурированные
    двумерные массивы строк с жесткой валидацией структуры.

    Attributes:
        model_name (str): Название целевой модели в локальном реестре Ollama.
    """

    def __init__(self, model_name: str = "llava") -> None:
        """Инициализирует экстрактор таблиц.

        Args:
            model_name (str): Имя модели Ollama. По умолчанию "llava".
        """
        self.model_name = model_name

    def extract_2d_array(self, image_path: str) -> List[List[str]]:
        """Извлекает двумерный массив строк из изображения таблицы.

        Метод осуществляет деструктуризацию ответа VLM, принудительно
        приводит типы данных к строкам и восстанавливает мерность массива
        при синтаксических сбоях генерации модели.

        Args:
            image_path (str): Путь к PNG-файлу сгенерированного кропа таблицы.

        Returns:
            List[List[str]]: Валидный двумерный массив строк для поля Table.data.
                В случае критической ошибки или сбоя VLM возвращает [].
        """
        if not os.path.exists(image_path):
            logger.error("[VLM] Файл изображения не найден: %s", image_path)
            return []

        prompt = (
            "Extract tabular data from this image into a JSON 2D array. "
            "STRICT RULES:\n"
            "1. Output ONLY a valid JSON object with key 'data'.\n"
            "2. EVERY value in the list MUST be a STRING (wrap numbers in quotes).\n"
            "3. NO markdown, NO explanations, NO ```json blocks."
        )

        try:
            logger.info("[VLM] Анализ структуры таблицы: %s", os.path.basename(image_path))
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                format="json",
                options={"temperature": 0.0},
            )

            raw_content: str = response["message"]["content"].strip()
            cleaned: str = raw_content.strip('`').removeprefix('json').strip()
            parsed_json: Dict[str, Any] = json.loads(cleaned)

            if isinstance(parsed_json, dict) and "data" in parsed_json:
                data_layer = parsed_json["data"]
                if isinstance(data_layer, list):
                    if data_layer and not isinstance(data_layer[0], list):
                        logger.warning("[VLM] Обнаружен плоский 1D-массив. Реструктуризация.")
                        parsed_json["data"] = [[str(item)] for item in data_layer]
                    else:
                        parsed_json["data"] = [
                            [str(cell) for cell in row]
                            if isinstance(row, list) else [str(row)]
                            for row in data_layer
                        ]
                else:
                    parsed_json["data"] = [[str(data_layer)]]
            else:
                parsed_json = {"data": []}

            validated_data = TableDataResponse(**parsed_json)
            logger.info("[VLM] Данные таблицы успешно верифицированы.")
            return validated_data.data

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning("[VLM] Ошибка валидации структуры данных: %s", e)
            return []
        except Exception as e:
            logger.error("[VLM] Ошибка локального сервера Ollama (Код: %s)", e)
            return []


class SpatialExtractor:
    """Оркестратор пространственной разметки и сегментации визуальных элементов.

    Использует DocLayout-YOLOv10 для локализации BBox объектов на страницах PDF,
    вырезает графические элементы через PyMuPDF и маршрутизирует таблицы в VLM.

    Attributes:
        output_img_dir (str): Каталог для персистентного сохранения изображений.
        vlm (VLMTableExtractor): Компонент мультимодального парсинга таблиц.
        model (YOLOv10): Модель глубокого обучения для детекции макета.
    """

    def __init__(self, output_img_dir: str = "images") -> None:
        """Инициализирует SpatialExtractor и загружает необходимые веса.

        Args:
            output_img_dir (str): Папка для сохранения графических артефактов.
        """
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)

        self.vlm = VLMTableExtractor(model_name="llava")

        logger.info("[YOLO] Загрузка предобученных весов DocLayout-YOLOv10...")
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(model_path)
        logger.info("[YOLO] Инициализация модели успешно завершена.")

    def extract_visuals(self, pdf_path: str) -> Tuple[List[Figure], List[Table]]:
        """Извлекает и валидирует все фигуры и таблицы из целевого PDF.

        Args:
            pdf_path (str): Путь к обрабатываемому PDF-документу.

        Returns:
            Tuple[List[Figure], List[Table]]: Списки валидированных Pydantic-моделей
                детекции фигур и табличных данных.
        """
        doc = fitz.open(pdf_path)
        figures_list: List[Figure] = []
        tables_list: List[Table] = []
        fig_cnt: int = 1
        tab_cnt: int = 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            temp_img_path = os.path.join(
                self.output_img_dir, f"temp_p{page_num}.png"
            )
            page.get_pixmap(dpi=72).save(temp_img_path)

            results = self.model.predict(
                temp_img_path, imgsz=1024, conf=0.25, verbose=False
            )
            
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            boxes = results[0].boxes
            if len(boxes) == 0:
                continue

            for box in boxes:
                class_name = self.model.names[int(box.cls[0])].lower()
                if class_name not in ["figure", "image", "table"]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_obj = BBox(
                    left=round(x1, 2),
                    top=round(y1, 2),
                    right=round(x2, 2),
                    bottom=round(y2, 2),
                )
                rect = fitz.Rect(x1, y1, x2, y2)

                if class_name in ["figure", "image"]:
                    img_filename = f"fig_{fig_cnt}_p{page_num + 1}.png"
                    path = os.path.join(self.output_img_dir, img_filename)
                    page.get_pixmap(clip=rect, dpi=150).save(path)

                    root_panel = Panel(bbox=bbox_obj, img_path=path)
                    fig_obj = Figure(
                        id=f"Figure {fig_cnt}",
                        caption="",
                        page=page_num + 1,
                        bbox=bbox_obj,
                        img_path=path,
                        panels=[root_panel],
                    )
                    figures_list.append(fig_obj)
                    fig_cnt += 1

                elif class_name == "table":
                    img_filename = f"tab_{tab_cnt}_p{page_num + 1}.png"
                    path = os.path.join(self.output_img_dir, img_filename)
                    
                    page.get_pixmap(clip=rect, dpi=300).save(path)

                    table_data = self.vlm.extract_2d_array(path)

                    tab_obj = Table(
                        id=f"Table {tab_cnt}",
                        caption="",
                        page=page_num + 1,
                        bbox=bbox_obj,
                        img_path=path,
                        data=table_data,
                    )
                    tables_list.append(tab_obj)
                    tab_cnt += 1

        doc.close()
        return figures_list, tables_list