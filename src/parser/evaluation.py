"""Оценка качества извлечения на отложенной выборке: precision / recall / F1.

Сравнивает предсказанный парсером результат с эталонной («золотой») разметкой
и считает метрики по основным полям итогового JSON (ТЗ, доп. баллы 8.3.6).
Покрываются не только метаданные, но и остальные части документа:

- Скалярные поля (title/journal/abstract): нечёткое сравнение (вложенность строк);
  doi/year — строгое. precision = доля верных среди заполненных нами,
  recall = доля верных среди тех, где эталон непуст.
- Множественные поля (authors/keywords/sections): per-doc P/R/F1 по множествам
  (фамилии авторов, нормализованные ключевые слова, заголовки секций) с
  macro-усреднением по документам.
- Счётные поля (figures/tables/equations): сравнение количества извлечённых
  элементов с эталоном (recall = найдено/эталон, precision = найдено/предсказано).

Поле попадает в отчёт, если оно базовое (title/doi/year/journal/authors) либо
если эталон содержит для него непустое значение хотя бы в одном документе —
поэтому частичная gold-разметка (например, только метаданные) тоже работает.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

#: Скалярные поля и режим сравнения:
#: "exact" — строгое равенство, "fuzzy" — вложенность строк,
#: "ratio" — похожесть текста выше порога (для длинных полей вроде abstract).
SCALAR_FIELDS: dict[str, str] = {
    "title": "fuzzy",
    "doi": "exact",
    "year": "exact",
    "journal": "fuzzy",
    "abstract": "ratio",
}

#: Порог похожести для длинного текста (abstract): доля совпадения 0..1.
TEXT_RATIO_THRESHOLD = 0.85

#: Множественные поля -> функция нормализации одного элемента.
SET_FIELDS: dict[str, Callable[[str], str]] = {}

#: Счётные поля (сравнение количества элементов).
COUNT_FIELDS: tuple[str, ...] = ("figures", "tables", "equations")

#: Поля, которые показываем всегда (даже при нулевом support) — для совместимости.
CORE_FIELDS: frozenset[str] = frozenset(
    {"title", "doi", "year", "journal", "authors"}
)


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


#: authors сравниваем по фамилиям, keywords/sections — по полной нормализации.
SET_FIELDS.update(
    {
        "authors": _surname,
        "keywords": _norm,
        "sections": _norm,
    }
)


def scalar_match(pred: Any, gold: Any, *, fuzzy: bool) -> bool:
    """Совпадение скалярного поля. fuzzy=True допускает вложенность строк."""
    np, ng = _norm(pred), _norm(gold)
    if not np or not ng:
        return False
    if np == ng:
        return True
    return fuzzy and (np in ng or ng in np)


def text_ratio_match(pred: Any, gold: Any, threshold: float) -> bool:
    """Совпадение длинного текста по похожести (для abstract).

    Строгая вложенность для свободного текста слишком хрупка (один символ ломает
    substring), поэтому сравниваем долю совпадения последовательностей.
    """
    np, ng = _norm(pred), _norm(gold)
    if not np or not ng:
        return False
    if np == ng or np in ng or ng in np:
        return True
    return SequenceMatcher(None, np, ng).ratio() >= threshold


def set_prf(
    pred: Iterable[str], gold: Iterable[str], normalizer: Callable[[str], str]
) -> tuple[float, float, float]:
    """Precision/Recall/F1 по множествам элементов одного документа."""
    pred_set = {n for n in (normalizer(x) for x in pred) if n}
    gold_set = {n for n in (normalizer(x) for x in gold) if n}
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


def authors_prf(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    """Precision/Recall/F1 по множествам фамилий авторов одного документа."""
    return set_prf(pred, gold, _surname)


def section_headings(sections: Any) -> list[str]:
    """Рекурсивно собирает заголовки из дерева секций (list[Section|dict])."""
    headings: list[str] = []
    for sec in sections or []:
        heading = sec.get("heading") if isinstance(sec, dict) else getattr(
            sec, "heading", None
        )
        if heading:
            headings.append(str(heading))
        subs = (
            sec.get("subsections")
            if isinstance(sec, dict)
            else getattr(sec, "subsections", None)
        )
        headings.extend(section_headings(subs))
    return headings


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


def _get(obj: Any, name: str) -> Any:
    """Достаёт поле из dict или pydantic-объекта."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _is_empty(value: Any) -> bool:
    """Пустое значение эталона/предсказания (None, '', [], 0 для счётчиков)."""
    return value in (None, "", [], {})


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom == 0 else 2 * precision * recall / denom


def _scalar_match_mode(pred: Any, gold: Any, mode: str) -> bool:
    """Совпадение скалярного поля по режиму exact/fuzzy/ratio."""
    if mode == "ratio":
        return text_ratio_match(pred, gold, TEXT_RATIO_THRESHOLD)
    return scalar_match(pred, gold, fuzzy=(mode == "fuzzy"))


def _score_scalar(predictions: list[Any], golds: list[Any], name: str) -> FieldScore:
    mode = SCALAR_FIELDS[name]
    predicted = correct = support = 0
    for pred, gold in zip(predictions, golds):
        pv, gv = _get(pred, name), _get(gold, name)
        if not _is_empty(gv):
            support += 1
        if not _is_empty(pv):
            predicted += 1
            if _scalar_match_mode(pv, gv, mode):
                correct += 1
    precision = correct / predicted if predicted else 0.0
    recall = correct / support if support else 0.0
    return FieldScore(precision, recall, _f1(precision, recall), support)


def _score_set(predictions: list[Any], golds: list[Any], name: str) -> FieldScore:
    normalizer = SET_FIELDS[name]
    sums = [0.0, 0.0, 0.0]
    support = 0
    for pred, gold in zip(predictions, golds):
        gold_items = _get(gold, name) or []
        if not gold_items:
            continue
        support += 1
        p, r, f = set_prf(_get(pred, name) or [], gold_items, normalizer)
        sums[0] += p
        sums[1] += r
        sums[2] += f
    if not support:
        return FieldScore(0.0, 0.0, 0.0, 0)
    return FieldScore(sums[0] / support, sums[1] / support, sums[2] / support, support)


def _score_count(predictions: list[Any], golds: list[Any], name: str) -> FieldScore:
    sums = [0.0, 0.0, 0.0]
    support = 0
    for pred, gold in zip(predictions, golds):
        gv = _get(gold, name)
        if gv is None:
            continue
        gold_n = int(gv)
        if gold_n <= 0:
            continue
        support += 1
        pred_n = int(_get(pred, name) or 0)
        tp = min(pred_n, gold_n)
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / gold_n
        sums[0] += precision
        sums[1] += recall
        sums[2] += _f1(precision, recall)
    if not support:
        return FieldScore(0.0, 0.0, 0.0, 0)
    return FieldScore(sums[0] / support, sums[1] / support, sums[2] / support, support)


def evaluate(predictions: list[Any], golds: list[Any]) -> Report:
    """Считает метрики по корпусу.

    Args:
        predictions: список предсказаний (dict или объект с нужными атрибутами).
        golds: список эталонов в том же порядке.

    Returns:
        :class:`Report` с метриками по каждому полю. Поле включается, если оно
        базовое (CORE_FIELDS) либо имеет ненулевой support в эталоне.
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions и golds должны быть одной длины")

    report = Report(documents=len(golds))

    scorers: list[tuple[str, FieldScore]] = []
    for name in SCALAR_FIELDS:
        scorers.append((name, _score_scalar(predictions, golds, name)))
    for name in SET_FIELDS:
        scorers.append((name, _score_set(predictions, golds, name)))
    for name in COUNT_FIELDS:
        scorers.append((name, _score_count(predictions, golds, name)))

    for name, score in scorers:
        if name in CORE_FIELDS or score.support > 0:
            report.fields[name] = score

    return report
