"""
Модуль извлечения визуальных элементов (Таблиц и Фигур).

Архитектура:
1. DocLayout-YOLOv10 - пространственная детекция объектов.
2. PyMuPDF (fitz) - прецизионный кроп изображений (dpi=300 для OCR).
3. LLaVA (через Ollama) - извлечение структурированных данных таблиц.
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
    """Извлекатель табличных данных на базе локальной VLM (LLaVA).

    Attributes:
        model_name (str): Имя модели Ollama.
    """

    def __init__(self, model_name: str = "llava") -> None:
        """Инициализирует экстрактор."""
        self.model_name = model_name

    def extract_2d_array(self, image_path: str) -> List[List[str]]:
        """Извлекает 2D-массив данных из изображения таблицы."""
        if not os.path.exists(image_path):
            return []

        prompt = (
            "Extract tabular data from this image into a JSON 2D array. "
            "STRICT RULES:\n"
            "1. Output ONLY a valid JSON object with key 'data'.\n"
            "2. Value MUST be a list of lists of STRINGS: [['str', 'str']].\n"
            "3. NO markdown, NO explanations, NO ```json blocks."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                format="json",
                options={"temperature": 0.0},
            )

            raw_content = response["message"]["content"].strip()
            cleaned = raw_content.strip("`").removeprefix("json").strip()

            parsed_json = json.loads(cleaned)
            return TableDataResponse(**parsed_json).data

        except (ValidationError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"[VLM] Ошибка извлечения: {e}")
            return []


class SpatialExtractor:
    """Оркестратор извлечения визуальных элементов."""

    def __init__(self, output_img_dir: str = "images") -> None:
        """Инициализирует экстрактор и модель YOLO."""
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)

        self.vlm = VLMTableExtractor(model_name="llava")

        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(model_path)

    def extract_visuals(self, pdf_path: str) -> Tuple[List[Figure], List[Table]]:
        """Главный метод извлечения фигур и таблиц."""
        doc = fitz.open(pdf_path)
        figures, tables = [], []
        fig_cnt, tab_cnt = 1, 1

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Временный кроп для YOLO
            temp_img = f"temp_p{page_num}.png"
            page.get_pixmap(dpi=72).save(temp_img)

            results = self.model.predict(temp_img, imgsz=1024, conf=0.25, verbose=False)
            os.remove(temp_img)

            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls[0])].lower()
                if class_name not in ["figure", "image", "table"]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = BBox(
                    left=round(x1, 2),
                    top=round(y1, 2),
                    right=round(x2, 2),
                    bottom=round(y2, 2),
                )
                rect = fitz.Rect(x1, y1, x2, y2)

                if class_name in ["figure", "image"]:
                    path = os.path.join(
                        self.output_img_dir, f"fig_{fig_cnt}_p{page_num + 1}.png"
                    )
                    page.get_pixmap(clip=rect, dpi=150).save(path)
                    figures.append(
                        Figure(
                            id=f"Figure {fig_cnt}",
                            caption="",
                            page=page_num + 1,
                            bbox=bbox,
                            img_path=path,
                            panels=[Panel(bbox=bbox, img_path=path)],
                        )
                    )
                    fig_cnt += 1
                else:
                    path = os.path.join(
                        self.output_img_dir, f"tab_{tab_cnt}_p{page_num + 1}.png"
                    )
                    page.get_pixmap(clip=rect, dpi=300).save(
                        path
                    )  # 300 DPI для качества!
                    tables.append(
                        Table(
                            id=f"Table {tab_cnt}",
                            caption="",
                            page=page_num + 1,
                            bbox=bbox,
                            img_path=path,
                            data=self.vlm.extract_2d_array(path),
                        )
                    )
                    tab_cnt += 1

        doc.close()
        return figures, tables
