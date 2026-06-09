"""Метрики качества по всему документу на отложенной выборке (ТЗ 8.3.6).

В отличие от ``eval_metadata.py`` (только метаданные «на лету»), этот скрипт
оценивает ПОЛНЫЙ результат пайплайна: читает готовые ``<имя>.json`` (объекты
Document, сгенерированные ``python -m parser``) и сравнивает с эталоном
``evaluation/gold.json`` по основным полям JSON — метаданные, abstract,
keywords, заголовки секций и количество фигур/таблиц/уравнений.

Сначала прогоняем парсер, затем считаем метрики по его выходу:

    uv run python -m parser --input <pdf-папка> --output out/
    uv run python scripts/eval_pipeline.py --pred-dir out/ --gold evaluation/gold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parser.evaluation import evaluate, section_headings


def _prediction_from_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Уплощает Document-JSON в плоский dict полей для метрик."""
    meta = doc.get("metadata", {})
    return {
        "title": meta.get("title"),
        "authors": meta.get("authors", []),
        "doi": meta.get("doi"),
        "year": meta.get("year"),
        "journal": meta.get("journal"),
        "abstract": meta.get("abstract"),
        "keywords": meta.get("keywords", []),
        "sections": section_headings(doc.get("sections", [])),
        "figures": len(doc.get("figures", [])),
        "tables": len(doc.get("tables", [])),
        "equations": len(doc.get("equations", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики качества по всему документу")
    parser.add_argument(
        "--pred-dir", required=True, help="Папка с JSON-выходами парсера"
    )
    parser.add_argument("--gold", default="evaluation/gold.json", help="Эталон JSON")
    parser.add_argument("--out", default="", help="Куда сохранить отчёт JSON (опц.)")
    args = parser.parse_args()

    gold_map: dict[str, dict[str, Any]] = json.loads(
        Path(args.gold).read_text("utf-8")
    )
    pred_dir = Path(args.pred_dir)

    preds: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    for filename, gold in gold_map.items():
        json_path = pred_dir / f"{Path(filename).stem}.json"
        if not json_path.exists():
            print(f"⚠ пропуск (нет выхода парсера): {json_path.name}")
            continue
        doc = json.loads(json_path.read_text("utf-8"))
        preds.append(_prediction_from_document(doc))
        golds.append(gold)

    if not preds:
        print("Нет ни одного сопоставления pred/gold — нечего считать.")
        return

    report = evaluate(preds, golds)

    print(f"\nДокументов: {report.documents}\n")
    header = f"{'поле':<12} {'precision':>10} {'recall':>10} {'f1':>8} {'support':>8}"
    print(header)
    print("-" * len(header))
    for name, score in report.fields.items():
        print(
            f"{name:<12} {score.precision:>10.3f} {score.recall:>10.3f} "
            f"{score.f1:>8.3f} {score.support:>8}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), "utf-8"
        )
        print(f"\nОтчёт сохранён: {args.out}")


if __name__ == "__main__":
    main()
