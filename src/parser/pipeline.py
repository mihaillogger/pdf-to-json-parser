import concurrent.futures
from pathlib import Path

import fitz
from loguru import logger

from parser.equations import EquationExtractor
from parser.extractor import get_page_blocks
from parser.figures import SpatialExtractor
from parser.metadata import extract_metadata
from parser.schemas import Document, Equation, Figure, PageBlock, Table
from parser.sections import build_section_tree


def enrich_equations_context(
    equations: list[Equation], blocks: list[PageBlock]
) -> None:
    """
    Обогащает уравнения контекстом. Сопоставляет Y-координаты
    ограничивающих рамок (BBox) формул и текстовых блоков Матвея.
    """
    for eq in equations:
        if not eq.bbox or not eq.page:
            continue

        page_blocks = [
            b
            for b in blocks
            if b.page_number == eq.page
            and getattr(b, "block_type", "text") == "text"
            and b.text
        ]

        if not page_blocks:
            continue

        above = [b for b in page_blocks if b.bbox.bottom <= eq.bbox.top + 10]
        below = [b for b in page_blocks if b.bbox.top >= eq.bbox.bottom - 10]

        context_parts = []
        if above:
            nearest_above = max(above, key=lambda b: b.bbox.bottom)
            context_parts.append((nearest_above.text or "").strip())

        if below:
            nearest_below = min(below, key=lambda b: b.bbox.top)
            context_parts.append((nearest_below.text or "").strip())

        if context_parts:
            eq.context = "\n[FORMULA PLACEHOLDER]\n".join(context_parts)


def process_single_file(
    pdf_path: Path,
    output_dir: Path,
    overwrite: bool,
    offline: bool = False,
    use_crossref: bool = True,
    use_llm: bool = True,
    extract_images: bool = True,
) -> None:
    """Полный цикл парсинга одного PDF документа."""
    json_path = output_dir / f"{pdf_path.stem}.json"

    if json_path.exists() and not overwrite:
        logger.info(f"Пропуск {pdf_path.name}: JSON уже существует.")
        return

    logger.info(f"Начало обработки: {pdf_path.name}")

    try:
        # 1. Базовый I/O
        blocks = get_page_blocks(str(pdf_path))
        if not blocks:
            logger.error(f"Файл {pdf_path.name} пуст или не содержит текста.")
            return

        # raw_text для JSON-поля (ТЗ 4.8): колоночный порядок из блоков — чище,
        # без артефактов вёрстки.
        raw_text = "\n".join(
            [
                b.text
                for b in blocks
                if getattr(b, "block_type", "text") == "text" and b.text
            ]
        )

        # Для поиска DOI/метаданных нужен ПОЛНЫЙ текст, включая колонтитулы:
        # DOI самой статьи обычно напечатан в шапке/футере, которые extractor
        # срезает из блоков (иначе find_doi не находит свой DOI или берёт чужой).
        try:
            with fitz.open(str(pdf_path)) as doc_full:
                meta_text = "\n".join(page.get_text() for page in doc_full)
        except Exception:
            meta_text = raw_text

        # 2. Метаданные
        logger.debug("Извлечение метаданных...")
        meta = extract_metadata(
            blocks=blocks,
            raw_text=meta_text,
            use_crossref=use_crossref,
            use_llm=use_llm,
            offline=offline,
        )

        # 3. Сборка дерева секций
        logger.debug("Построение иерархии секций...")
        section_tree = build_section_tree(blocks)

        # 4. Визуальные элементы: Фигуры и Таблицы
        figures_list: list[Figure] = []
        tables_list: list[Table] = []
        if extract_images:
            logger.debug("Запуск SpatialExtractor (YOLOv10 + LLaVA)...")
            img_dir = output_dir / "images" / pdf_path.stem
            img_dir.mkdir(parents=True, exist_ok=True)

            spatial = SpatialExtractor(output_img_dir=str(img_dir))
            figures_list, tables_list = spatial.extract_visuals(str(pdf_path))

        # 5. Уравнения (Твоя YOLOv8 + Pix2Tex)
        equations_list = []
        project_root = Path(__file__).resolve().parent.parent.parent
        eq_weights = project_root / "weights" / "best.pt"

        if eq_weights.exists():
            logger.debug("Запуск EquationExtractor...")
            eq_extractor = EquationExtractor(model_path=str(eq_weights))
            equations_list = eq_extractor.process_pdf(str(pdf_path))

            # === СТРОГАЯ ИНТЕГРАЦИЯ КОНТЕКСТА ===
            logger.debug("Обогащение формул текстовым контекстом...")
            enrich_equations_context(equations_list, blocks)
            # ====================================
        else:
            logger.warning(
                f"Веса для уравнений не найдены ({eq_weights}). "
                "Парсинг формул пропущен."
            )

        # 6. Финальная Pydantic-сборка
        logger.debug("Сборка итогового объекта Document...")
        doc = Document(
            metadata=meta,
            sections=section_tree,
            figures=figures_list,
            tables=tables_list,
            equations=equations_list,
            acknowledgments=None,  # Можно дописать поиск внутри section_tree
            raw_text=raw_text,
        )

        # 7. Дамп в JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

        logger.success(f"Документ успешно собран: {json_path.name}")

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке {pdf_path.name}: {e}")
        logger.exception("Полный traceback:")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    overwrite: bool,
    offline: bool = False,
    use_crossref: bool = True,
    use_llm: bool = True,
    extract_images: bool = True,
) -> None:
    """Пакетная обработка директории с параллельным выполнением."""
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"В директории {input_dir} не найдено PDF-файлов.")
        return

    logger.info(f"Найдено файлов: {len(pdf_files)}. Запуск {workers} воркеров.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_single_file,
                pdf,
                output_dir,
                overwrite,
                offline,
                use_crossref,
                use_llm,
                extract_images,
            )
            for pdf in pdf_files
        ]
        concurrent.futures.wait(futures)

    logger.info("Пакетная обработка директории завершена.")
