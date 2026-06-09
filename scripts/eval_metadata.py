"""Считает метрики качества метаданных на отложенной выборке.

Сравнивает извлечённые парсером метаданные с «золотой» разметкой
(``evaluation/gold.json``) и печатает precision/recall/F1 по полям.

Пример:
    uv run python scripts/eval_metadata.py --input . --gold evaluation/gold.json \\
        --mode offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fitz

from parser.evaluation import evaluate
from parser.extractor import get_page_blocks
from parser.metadata import extract_metadata

#: Режимы извлечения -> флаги extract_metadata.
MODES: dict[str, dict[str, bool]] = {
    "online": {"offline": False, "use_crossref": True, "use_llm": False},
    "offline": {"offline": True, "use_crossref": False, "use_llm": False},
    "offline-llm": {"offline": True, "use_crossref": False, "use_llm": True},
}


def _full_text(pdf_path: str) -> str:
    """Полный текст документа (с колонтитулами) для поиска DOI."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _predict(pdf_path: str, flags: dict[str, bool]) -> dict[str, Any]:
    """Извлекает метаданные одного PDF и приводит к dict нужных полей."""
    blocks = get_page_blocks(pdf_path)
    meta = extract_metadata(blocks, _full_text(pdf_path), **flags)
    return {
        "title": meta.title,
        "authors": meta.authors,
        "doi": meta.doi,
        "year": meta.year,
        "journal": meta.journal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики качества метаданных")
    parser.add_argument("--input", default=".", help="Папка с PDF")
    parser.add_argument("--gold", default="evaluation/gold.json", help="Эталон JSON")
    parser.add_argument("--mode", choices=list(MODES), default="offline")
    parser.add_argument("--out", default="", help="Куда сохранить отчёт JSON (опц.)")
    args = parser.parse_args()

    gold_map: dict[str, dict[str, Any]] = json.loads(Path(args.gold).read_text("utf-8"))
    flags = MODES[args.mode]
    input_dir = Path(args.input)

    preds: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    for filename, gold in gold_map.items():
        pdf_path = input_dir / filename
        if not pdf_path.exists():
            print(f"⚠ пропуск (нет файла): {filename}")
            continue
        preds.append(_predict(str(pdf_path), flags))
        golds.append(gold)

    report = evaluate(preds, golds)

    print(f"\nРежим: {args.mode} | документов: {report.documents}\n")
    header = f"{'поле':<10} {'precision':>10} {'recall':>10} {'f1':>8} {'support':>8}"
    print(header)
    print("-" * len(header))
    for name, score in report.fields.items():
        print(
            f"{name:<10} {score.precision:>10.3f} {score.recall:>10.3f} "
            f"{score.f1:>8.3f} {score.support:>8}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), "utf-8"
        )
        print(f"\nОтчёт сохранён: {args.out}")


if __name__ == "__main__":
    main()
