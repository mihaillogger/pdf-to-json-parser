import sys
import warnings
from pathlib import Path
from typing import Annotated

import fitz  # Импортируем fitz сюда только ради отключения логов
import typer
from loguru import logger

from parser import pipeline

warnings.filterwarnings("ignore")

fitz.TOOLS.mupdf_display_errors(False)

app = typer.Typer(
    help="Конвейер для парсинга научных PDF-статей в структурированный JSON.",
    add_completion=False,
)


@app.command()  # type: ignore[untyped-decorator]
def process(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Путь к PDF-файлу или директории с PDF-документами",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Директория для сохранения итоговых JSON и изображений",
        ),
    ],
    workers: Annotated[
        int,
        typer.Option("--workers", help="Количество параллельных процессов"),
    ] = 1,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Перезаписывать существующие JSON-файлы",
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Уровень логирования (INFO, DEBUG)"),
    ] = "INFO",
    extract_images: Annotated[
        bool,
        typer.Option("--extract-images", help="Сохранять изображения фигур"),
    ] = True,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help=("Отключить все сетевые запросы (принудительный локальный режим)"),
        ),
    ] = False,
    use_crossref: Annotated[
        bool,
        typer.Option(
            "--crossref/--no-crossref",
            help="Использовать Crossref API для поиска метаданных",
        ),
    ] = True,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Использовать LLM для извлечения сложных структур",
        ),
    ] = True,
) -> None:
    """
    Главная точка входа. Валидирует пути и передает управление в ядро
    парсера (pipeline.py).
    """
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | {message}"
    )
    logger.add(sys.stdout, level=log_level.upper(), format=log_format)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(output_dir / "run.log", level="DEBUG", rotation="10 MB")

    logger.info(f"Запуск парсера. Вход: {input_path}")
    logger.info(
        f"Параметры: workers={workers}, overwrite={overwrite}, "
        f"extract_images={extract_images}"
    )
    logger.info(
        f"Режимы работы: offline={offline}, use_crossref={use_crossref}, "
        f"use_llm={use_llm}"
    )

    if not input_path.exists():
        logger.error(f"Указанный путь не существует: {input_path}")
        raise typer.Exit(code=1)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        logger.info("Режим: Одиночный документ")
        status = pipeline.process_single_file(
            pdf_path=input_path,
            output_dir=output_dir,
            overwrite=overwrite,
            offline=offline,
            use_crossref=use_crossref,
            use_llm=use_llm,
            extract_images=extract_images,
        )
        pipeline.log_doc_status(status)

    elif input_path.is_dir():
        logger.info("Режим: Пакетная обработка директории")
        pipeline.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            workers=workers,
            overwrite=overwrite,
            offline=offline,
            use_crossref=use_crossref,
            use_llm=use_llm,
            extract_images=extract_images,
        )

    else:
        logger.error("Указанный путь не является PDF-файлом или директорией.")
        raise typer.Exit(code=1)

    logger.info("Отработка интерфейса завершена.")


if __name__ == "__main__":
    app()
