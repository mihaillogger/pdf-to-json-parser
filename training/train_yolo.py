"""
Модуль для обучения кастомной модели YOLOv8.
Реализует загрузку датасета с Roboflow, запуск процесса тренировки
и автоматическое перемещение финальных весов в корневую директорию проекта.
"""

import os
import shutil
import logging
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Основной пайплайн загрузки данных, обучения и сохранения модели.

    Raises:
        ValueError: Если переменная окружения ROBOFLOW_API_KEY не задана.
        FileNotFoundError: Если после завершения обучения итоговый файл весов не найден.
    """
    root_dir = Path(__file__).resolve().parent.parent
    weights_dir = root_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    final_model_path = weights_dir / "best.pt"

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден. Задайте ROBOFLOW_API_KEY в .env")

    logger.info("Загрузка датасета с Roboflow.")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("s-workspace-yuyzr").project("lab-iflyz")
    version = project.version(1)
    dataset = version.download("yolov8")

    logger.info("Датасет загружен. Запуск обучения YOLOv8n.")
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,
        imgsz=1024,
        batch=8,
        device=0
    )
    
    yolo_save_dir = Path(model.trainer.save_dir)
    generated_weights_path = yolo_save_dir / "weights" / "best.pt"
    
    if not generated_weights_path.exists():
        raise FileNotFoundError(f"Файл весов не сгенерирован: {generated_weights_path}")
        
    shutil.copy(generated_weights_path, final_model_path)
    logger.info("Обучение завершено. Веса сохранены: %s", final_model_path)


if __name__ == "__main__":
    main()