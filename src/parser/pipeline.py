import concurrent.futures
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import fitz
from loguru import logger

from parser.equations import EquationExtractor
from parser.extractor import get_page_blocks
from parser.figures import SpatialExtractor
from parser.metadata import extract_metadata
from parser.schemas import BBox, Document, Equation, Figure, PageBlock, Table
from parser.sections import build_section_tree, extract_acknowledgments

#: Обязательные поля (ТЗ 8.1): если какое-то пусто — статус «частичный успех».
_REQUIRED_FIELDS = ("title", "authors", "abstract", "doi", "sections", "raw_text")


@dataclass
class DocStatus:
    """Результат обработки одного документа."""

    name: str
    status: str  # "success" | "partial" | "error" | "skipped"
    seconds: float = 0.0
    missing: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BatchSummary:
    """Агрегированная сводка по пакетной обработке."""

    total: int = 0
    success: int = 0
    partial: int = 0
    errors: int = 0
    skipped: int = 0
    total_seconds: float = 0.0
    avg_seconds: float = 0.0


def _missing_required(doc: Document) -> list[str]:
    """Возвращает список обязательных полей, оставшихся пустыми."""
    meta = doc.metadata
    values: dict[str, object] = {
        "title": (meta.title or "").strip(),
        "authors": meta.authors,
        "abstract": meta.abstract,
        "doi": meta.doi,
        "sections": doc.sections,
        "raw_text": (doc.raw_text or "").strip(),
    }
    return [name for name in _REQUIRED_FIELDS if not values[name]]


def log_doc_status(status: DocStatus) -> None:
    """Логирует статус по одному документу (ТЗ 5.3.1)."""
    if status.status == "success":
        logger.success(f"[OK] {status.name} ({status.seconds:.1f}s)")
    elif status.status == "partial":
        fields = ", ".join(status.missing)
        logger.warning(
            f"[ЧАСТИЧНО] {status.name} ({status.seconds:.1f}s) — пустые поля: {fields}"
        )
    elif status.status == "error":
        logger.error(f"[ОШИБКА] {status.name}: {status.error}")
    else:
        logger.info(f"[ПРОПУСК] {status.name}: JSON уже существует")


def summarize(results: list[DocStatus], total_seconds: float) -> BatchSummary:
    """Считает агрегаты по списку результатов (ТЗ 5.3.2)."""
    processed = [r for r in results if r.status in ("success", "partial", "error")]
    avg = sum(r.seconds for r in processed) / len(processed) if processed else 0.0
    return BatchSummary(
        total=len(results),
        success=sum(1 for r in results if r.status == "success"),
        partial=sum(1 for r in results if r.status == "partial"),
        errors=sum(1 for r in results if r.status == "error"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        total_seconds=total_seconds,
        avg_seconds=avg,
    )


def _log_summary(summary: BatchSummary) -> None:
    """Печатает финальную сводку пакетной обработки."""
    logger.info("=" * 60)
    logger.info(
        f"ИТОГ: всего={summary.total}, успешно={summary.success}, "
        f"частично={summary.partial}, ошибок={summary.errors}, "
        f"пропущено={summary.skipped}"
    )
    logger.info(
        f"Время: общее={summary.total_seconds:.1f}s, "
        f"среднее/документ={summary.avg_seconds:.1f}s"
    )
    logger.info("=" * 60)


#: Подпись к визуальному элементу: "Figure 1.", "Fig. S2", "Scheme 3",
#: "Table 4", а также рус. варианты ("Рис. 1", "Таблица 2", "Схема 1").
_CAPTION_RE = re.compile(
    r"^\s*(?P<label>figure|fig\.?|scheme|table|таблица|рисунок|рис\.?|схема)"
    r"\s*(?P<num>S?\d+[a-z]?)",
    re.IGNORECASE,
)

#: Канонизация метки подписи к виду из ТЗ (англ., с заглавной).
_LABEL_CANON: dict[str, str] = {
    "figure": "Figure",
    "fig": "Figure",
    "рис": "Figure",
    "рисунок": "Figure",
    "scheme": "Scheme",
    "схема": "Scheme",
    "table": "Table",
    "таблица": "Table",
}

#: Метки, относящиеся к «фигурам» (всё, что детектор отдаёт как figure/image).
_FIGURE_KINDS: frozenset[str] = frozenset({"Figure", "Scheme"})

#: Максимальное расстояние (в точках PDF) между элементом и его подписью.
_MAX_CAPTION_DISTANCE = 200.0


class _Captionable(Protocol):
    """Структурный протокол для Figure/Table: общие поля для матчинга подписи."""

    id: str
    caption: str
    page: int
    bbox: BBox


@dataclass
class _CaptionCandidate:
    """Кандидат-подпись, найденный в текстовом слое."""

    kind: str  # "Figure" | "Scheme" | "Table"
    id: str  # напр. "Figure S1"
    text: str
    page: int
    bbox: BBox
    used: bool = False


def _find_caption_candidates(blocks: list[PageBlock]) -> list[_CaptionCandidate]:
    """Находит в текстовых блоках строки-подписи вида «Figure N. …»/«Table N. …»."""
    candidates: list[_CaptionCandidate] = []
    for b in blocks:
        if b.block_type != "text" or not b.text:
            continue
        match = _CAPTION_RE.match(b.text)
        if not match:
            continue
        kind = _LABEL_CANON.get(match.group("label").lower().rstrip("."))
        if kind is None:
            continue
        num = match.group("num").upper()
        candidates.append(
            _CaptionCandidate(
                kind=kind,
                id=f"{kind} {num}",
                text=b.text,
                page=b.page_number,
                bbox=b.bbox,
            )
        )
    return candidates


def _caption_distance(v: BBox, c: BBox) -> float:
    """Близость подписи к элементу: вертикальный зазор + штраф за несовпадение по X.

    Подпись обычно лежит прямо под фигурой или над таблицей и горизонтально
    перекрывается с ней. Чем меньше значение — тем вероятнее, что это «своя» подпись.
    """
    h_overlap = min(v.right, c.right) - max(v.left, c.left)
    h_penalty = 0.0 if h_overlap > 0 else -h_overlap

    if c.top >= v.bottom:  # подпись ниже элемента
        v_gap = c.top - v.bottom
    elif c.bottom <= v.top:  # подпись выше элемента
        v_gap = v.top - c.bottom
    else:  # вертикально перекрываются
        v_gap = 0.0

    return v_gap + h_penalty


def _assign_captions(
    items: Sequence[_Captionable],
    candidates: list[_CaptionCandidate],
    kinds: frozenset[str],
) -> None:
    """Жадно привязывает к каждому элементу ближайшую неиспользованную подпись."""
    for item in items:
        best: _CaptionCandidate | None = None
        best_dist = _MAX_CAPTION_DISTANCE
        for cand in candidates:
            if cand.used or cand.kind not in kinds or cand.page != item.page:
                continue
            dist = _caption_distance(item.bbox, cand.bbox)
            if dist < best_dist:
                best, best_dist = cand, dist
        if best is not None:
            best.used = True
            item.id = best.id
            item.caption = best.text


def enrich_visual_captions(
    figures: list[Figure], tables: list[Table], blocks: list[PageBlock]
) -> None:
    """Проставляет фигурам/таблицам реальные ``id``/``caption`` из текста (ТЗ 4.4/4.5).

    Пространственный детектор отдаёт BBox и кропы, но не текст: объекты приходят
    с пустым ``caption`` и дефолтным ``id`` ("Figure 1"). Здесь мы находим в
    текстовом слое ближайшую подпись («Figure 2. …», «Table 1. …») и
    перезаписываем поля реальными значениями. Если подпись не найдена —
    дефолтные значения сохраняются.
    """
    candidates = _find_caption_candidates(blocks)
    if not candidates:
        return
    # Фигуры и схемы матчим из «фигурного» пула, таблицы — из «табличного».
    _assign_captions(figures, candidates, _FIGURE_KINDS)
    _assign_captions(tables, candidates, frozenset({"Table"}))


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
) -> DocStatus:
    """Полный цикл парсинга одного PDF документа.

    Returns:
        DocStatus: статус обработки (success/partial/error/skipped), список
        неизвлечённых обязательных полей и время обработки.
    """
    name = pdf_path.name
    start = time.perf_counter()
    json_path = output_dir / f"{pdf_path.stem}.json"

    if json_path.exists() and not overwrite:
        logger.info(f"Пропуск {name}: JSON уже существует.")
        return DocStatus(name=name, status="skipped")

    logger.info(f"Начало обработки: {name}")

    try:
        # 1. Базовый I/O
        blocks = get_page_blocks(str(pdf_path))
        if not blocks:
            logger.error(f"Файл {name} пуст или не содержит текста.")
            return DocStatus(
                name=name,
                status="error",
                seconds=time.perf_counter() - start,
                error="пустой документ или нет текстового слоя",
            )

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

            # Подписи и id берём из текстового слоя (детектор их не парсит).
            logger.debug("Сопоставление подписей фигур/таблиц с текстом...")
            enrich_visual_captions(figures_list, tables_list, blocks)

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

        # Вырезаем благодарности из дерева перед финальной сборкой
        logger.debug("Извлечение секции Acknowledgments...")
        ack_text = extract_acknowledgments(section_tree)

        # 6. Финальная Pydantic-сборка
        logger.debug("Сборка итогового объекта Document...")
        doc = Document(
            metadata=meta,
            sections=section_tree,
            figures=figures_list,
            tables=tables_list,
            equations=equations_list,
            acknowledgments=ack_text,
            raw_text=raw_text,
        )

        # 7. Дамп в JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

        elapsed = time.perf_counter() - start
        missing = _missing_required(doc)
        if missing:
            logger.warning(f"{name}: частичный успех, пустые поля: {missing}")
            return DocStatus(
                name=name, status="partial", seconds=elapsed, missing=missing
            )
        logger.success(f"Документ успешно собран: {json_path.name}")
        return DocStatus(name=name, status="success", seconds=elapsed)

    except Exception as e:
        # Падение одного документа не прерывает пакет (ТЗ 5.4); трейс — в run.log.
        logger.error(f"Критическая ошибка при обработке {name}: {e}")
        logger.exception("Полный traceback:")
        return DocStatus(
            name=name,
            status="error",
            seconds=time.perf_counter() - start,
            error=str(e),
        )


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

    results: list[DocStatus] = []
    batch_start = time.perf_counter()

    if workers <= 1:
        # Последовательно в главном процессе — статусы гарантированно идут в run.log.
        for pdf in pdf_files:
            status = process_single_file(
                pdf, output_dir, overwrite, offline, use_crossref, use_llm,
                extract_images,
            )
            log_doc_status(status)
            results.append(status)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_file, pdf, output_dir, overwrite, offline,
                    use_crossref, use_llm, extract_images,
                ): pdf
                for pdf in pdf_files
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    status = future.result()
                except Exception as exc:  # воркер упал на уровне процесса
                    pdf = futures[future]
                    status = DocStatus(name=pdf.name, status="error", error=str(exc))
                log_doc_status(status)
                results.append(status)

    _log_summary(summarize(results, time.perf_counter() - batch_start))
    logger.info("Пакетная обработка директории завершена.")
