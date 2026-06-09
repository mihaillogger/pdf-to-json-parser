"""Оценка качества извлечения метаданных: precision / recall / F1.

Сравнивает предсказанные метаданные с эталонными («золотая» разметка: CrossRef
по DOI + ручная проверка) и считает метрики по основным полям ТЗ:
title, authors, doi, year, journal.

- Скалярные поля (title/doi/year/journal): precision = доля верных среди
  заполненных нами; recall = доля верных среди тех, где эталон непуст.
- authors: множественное поле -> per-doc P/R/F1 по фамилиям, усреднение (macro).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

#: Поля, по которым считаем метрики.
SCALAR_FIELDS = ("title", "doi", "year", "journal")


def _norm(value: Any) -> str:
    """Нормализует значение для сравнения: без диакритики, регистра, пунктуации."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _surname(author: str) -> str:
    """Фамилия для сопоставления авторов (часть до запятой либо первый токен)."""
    head = author.split(",")[0] if "," in author else author.split(" ")[0]
    return _norm(head)


def scalar_match(pred: Any, gold: Any, *, fuzzy: bool) -> bool:
    """Совпадение скалярного поля. fuzzy=True допускает вложенность строк."""
    np, ng = _norm(pred), _norm(gold)
    if not np or not ng:
        return False
    if np == ng:
        return True
    return fuzzy and (np in ng or ng in np)


def authors_prf(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    """Precision/Recall/F1 по множествам фамилий авторов одного документа."""
    pred_set = {s for s in (_surname(a) for a in pred) if s}
    gold_set = {s for s in (_surname(a) for a in gold) if s}
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    denom = precision + recall
    f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
    return precision, recall, f1


@dataclass
class FieldScore:
    """Метрики одного поля по корпусу."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0  # документов, где эталонное значение непусто

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "support": self.support,
        }


@dataclass
class Report:
    """Итоговый отчёт по всем полям."""

    fields: dict[str, FieldScore] = field(default_factory=dict)
    documents: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "fields": {name: fs.as_dict() for name, fs in self.fields.items()},
        }


def _get(meta: Any, name: str) -> Any:
    """Достаёт поле из dict или pydantic-объекта Metadata."""
    return meta.get(name) if isinstance(meta, dict) else getattr(meta, name, None)


def evaluate(predictions: list[Any], golds: list[Any]) -> Report:
    """Считает метрики по корпусу.

    Args:
        predictions: список предсказанных метаданных (dict или Metadata).
        golds: список эталонных метаданных в том же порядке.

    Returns:
        :class:`Report` с метриками по каждому полю.
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions и golds должны быть одной длины")

    report = Report(documents=len(golds))

    for fname in SCALAR_FIELDS:
        predicted = correct = support = 0
        fuzzy = fname in ("title", "journal")
        for pred, gold in zip(predictions, golds):
            pv, gv = _get(pred, fname), _get(gold, fname)
            if gv not in (None, "", []):
                support += 1
            if pv not in (None, "", []):
                predicted += 1
                if scalar_match(pv, gv, fuzzy=fuzzy):
                    correct += 1
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        report.fields[fname] = FieldScore(precision, recall, f1, support)

    # authors — macro-усреднение по документам, где эталон непуст.
    sums = [0.0, 0.0, 0.0]
    support = 0
    for pred, gold in zip(predictions, golds):
        gold_authors = _get(gold, "authors") or []
        if not gold_authors:
            continue
        support += 1
        p, r, f = authors_prf(_get(pred, "authors") or [], gold_authors)
        sums[0] += p
        sums[1] += r
        sums[2] += f
    if support:
        report.fields["authors"] = FieldScore(
            sums[0] / support, sums[1] / support, sums[2] / support, support
        )
    else:
        report.fields["authors"] = FieldScore(0.0, 0.0, 0.0, 0)

    return report
