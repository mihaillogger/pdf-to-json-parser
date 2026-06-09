"""
Модуль извлечения визуальных элементов (Таблиц и Фигур).

Архитектура:
1. DocLayout-YOLOv10 - пространственная детекция BBox.
2. PyMuPDF (fitz) - физический кроп изображений (dpi=300 для OCR таблиц).
3. Qwen2.5-VL (через Ollama) - извлечение 2D-массивов таблиц.
"""

import json
import logging
import os
from typing import List, Tuple

import fitz
import ollama
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from pydantic import ValidationError

from parser.schemas import BBox, Figure, Panel, Table, TableDataResponse

logger = logging.getLogger(__name__)


class VLMTableExtractor:
    """Извлекатель табличных данных на базе локальной VLM.

    Использует мультимодальную нейросеть для преобразования изображений
    таблиц в структурированный JSON 2D-массив.

    Attributes:
        model_name (str): Название локальной модели для Ollama.
    """

    def __init__(self, model_name: str = "qwen2.5-vl") -> None:
        """Инициализирует экстрактор с указанной моделью.

        Args:
            model_name (str): Имя модели Ollama (по умолчанию "qwen2.5-vl").
        """
        self.model_name = model_name

    def extract_2d_array(self, image_path: str) -> List[List[str]]:
        """Извлекает данные из изображения таблицы.

        Args:
            image_path (str): Путь к PNG/JPG файлу таблицы.

        Returns:
            List[List[str]]: Двумерный массив строк. Если извлечение не удалось,
            возвращает пустой список.
        """
        if not os.path.exists(image_path):
            return []

        prompt = (
            "Extract tabular data from this image into a JSON 2D array. "
            "STRICT RULES:\n"
            "1. Output ONLY a valid JSON object containing a single key 'data'.\n"
            "2. The value MUST be a list of lists of STRINGS: [['str', 'str'], ['str', 'str']].\n"
            "3. If a cell contains a number, wrap it in quotes (e.g., '18', not 18).\n"
            "4. Absolutely NO markdown formatting, NO explanations, NO ```json blocks."
        )

        try:
            logger.info("[VLM] Анализ таблицы: %s", os.path.basename(image_path))
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                format="json",
                options={"temperature": 0.0},
            )

            raw_content = response["message"]["content"].strip()
            
            # Принудительная зачистка от маркдауна, если модель нарушила промпт
            cleaned_content = raw_content.strip('`').removeprefix('json').strip()
            
            logger.debug("[VLM] Сырой вывод модели: %s", cleaned_content)
            parsed_json = json.loads(cleaned_content)

            # Строгая валидация через Pydantic
            validated_data = TableDataResponse(**parsed_json)
            logger.info("[VLM] Успех: 2D массив получен.")
            return validated_data.data

        except ValidationError as e:
            logger.warning(
                "[VLM] Ошибка валидации структуры (кривой формат):\n%s\nСырой вывод: %s", 
                e, raw_content
            )
            return []
        except json.JSONDecodeError as e:
            logger.warning(
                "[VLM] Нейросеть вернула невалидный JSON: %s\nСырой вывод: %s", 
                e, raw_content
            )
            return []
        except Exception as e:
            logger.error("[VLM] Системная ошибка Ollama: %s", e)
            return []


class SpatialExtractor:
    """Главный оркестратор пространственной геометрии и извлечения визуальных элементов.

    Отвечает за детекцию BBox через YOLO, вырезание изображений через PyMuPDF
    и маршрутизацию таблиц в VLM-экстрактор.

    Attributes:
        output_img_dir (str): Директория для сохранения вырезанных картинок.
        vlm (VLMTableExtractor): Инстанс VLM для распознавания таблиц.
        model (YOLOv10): Загруженная модель детекции макета.
    """

    def __init__(self, output_img_dir: str = "images") -> None:
        """Инициализирует оркестратор и загружает веса YOLO.

        Args:
            output_img_dir (str): Папка для сохранения изображений.
        """
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)

        self.vlm = VLMTableExtractor(model_name="qwen2.5-vl")

        logger.info("Загрузка весов DocLayout-YOLO")
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(model_path)
        logger.info("Модель YOLO успешно загружена.")

    def extract_visuals(self, pdf_path: str) -> Tuple[List[Figure], List[Table]]:
        """Извлекает фигуры и таблицы из PDF файла.

        Args:
            pdf_path (str): Путь к анализируемому PDF документу.

        Returns:
            Tuple[List[Figure], List[Table]]: Кортеж из двух списков, содержащих
            валидированные Pydantic-объекты Figure и Table.
        """
        doc = fitz.open(pdf_path)
        figures_list: List[Figure] = []
        tables_list: List[Table] = []

        fig_counter = 1
        tab_counter = 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            temp_img_path = os.path.join(
                self.output_img_dir, f"temp_yolo_p{page_num}.png"
            )
            page.get_pixmap(dpi=72).save(temp_img_path)

            results = self.model.predict(
                temp_img_path, imgsz=1024, conf=0.25, verbose=False
            )
            boxes = results[0].boxes
            names = self.model.names

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            if len(boxes) == 0:
                continue

            for box in boxes:
                class_name = names[int(box.cls[0])].lower()

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
                    pix = page.get_pixmap(clip=rect, dpi=150)
                    img_filename = f"figure_{fig_counter}_p{page_num + 1}.png"
                    img_path = os.path.join(self.output_img_dir, img_filename)
                    pix.save(img_path)

                    root_panel = Panel(bbox=bbox_obj, img_path=img_path)
                    fig_obj = Figure(
                        id=f"Figure {fig_counter}",
                        caption="",
                        page=page_num + 1,
                        bbox=bbox_obj,
                        img_path=img_path,
                        panels=[root_panel],
                    )
                    figures_list.append(fig_obj)
                    fig_counter += 1

                elif class_name == "table":
                    # Жесткое требование архитектуры: 300 DPI для VLM
                    pix = page.get_pixmap(clip=rect, dpi=300)
                    img_filename = f"table_{tab_counter}_p{page_num + 1}.png"
                    img_path = os.path.join(self.output_img_dir, img_filename)
                    pix.save(img_path)

                    table_data = self.vlm.extract_2d_array(img_path)

                    tab_obj = Table(
                        id=f"Table {tab_counter}",
                        caption="",
                        page=page_num + 1,
                        bbox=bbox_obj,
                        img_path=img_path,
                        data=table_data,
                    )
                    tables_list.append(tab_obj)
                    tab_counter += 1

        doc.close()
        return figures_list, tables_list