"""Тесты модуля метрик качества (parser.evaluation)."""

from __future__ import annotations

from parser.evaluation import (
    authors_prf,
    evaluate,
    scalar_match,
    section_headings,
    set_prf,
)


def test_scalar_match_exact_and_normalized() -> None:
    assert scalar_match("Hello World", "hello  world", fuzzy=False)
    assert scalar_match("10.1016/J.X", "10.1016/j.x", fuzzy=False)
    assert not scalar_match(None, "x", fuzzy=True)
    assert not scalar_match("x", "", fuzzy=True)


def test_scalar_match_fuzzy_containment() -> None:
    # Заголовок-обрезок засчитывается при fuzzy (вложенность).
    assert scalar_match(
        "A novel image analysis methodology",
        "A novel image analysis methodology for online monitoring",
        fuzzy=True,
    )
    assert not scalar_match(
        "A novel image analysis methodology",
        "A novel image analysis methodology for online monitoring",
        fuzzy=False,
    )


def test_authors_prf_perfect() -> None:
    p, r, f = authors_prf(["Yao, Yunjin", "Liu, Yating"], ["Yao, Y.", "Liu, Y."])
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_authors_prf_partial() -> None:
    # Предсказали 1 из 2 авторов, без лишних -> P=1.0, R=0.5.
    p, r, _ = authors_prf(["Yao, Yunjin"], ["Yao, Y.", "Liu, Y."])
    assert p == 1.0
    assert r == 0.5


def test_authors_prf_with_false_positive() -> None:
    # 1 верный + 1 лишний -> P=0.5, R=1.0.
    p, r, _ = authors_prf(["Yao, Yunjin", "Ghost, Person"], ["Yao, Y."])
    assert p == 0.5
    assert r == 1.0


def test_evaluate_report_scalars_and_authors() -> None:
    preds = [
        {
            "title": "A Study of Catalysts",
            "doi": "10.1/x",
            "year": 2025,
            "journal": "Chem Sci",
            "authors": ["Smith, John"],
        },
        {
            "title": "Wrong Title",
            "doi": None,
            "year": 2020,
            "journal": None,
            "authors": [],
        },
    ]
    golds = [
        {
            "title": "A Study of Catalysts",
            "doi": "10.1/x",
            "year": 2025,
            "journal": "Chemical Science",
            "authors": ["Smith, J."],
        },
        {
            "title": "Real Second Title",
            "doi": "10.2/y",
            "year": 2020,
            "journal": "Nature",
            "authors": ["Doe, Jane"],
        },
    ]

    report = evaluate(preds, golds)

    assert report.documents == 2
    doi = report.fields["doi"]
    # doi: заполнили 1, верный 1 -> P=1.0; эталонов 2 -> R=0.5
    assert doi.precision == 1.0
    assert doi.recall == 0.5
    # year: оба верные -> P=R=1.0
    assert report.fields["year"].precision == 1.0
    assert report.fields["year"].recall == 1.0
    # authors: doc1 идеально, doc2 пусто (0) -> macro recall = 0.5
    assert report.fields["authors"].support == 2
    assert report.fields["authors"].recall == 0.5


def test_evaluate_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        evaluate([{}], [{}, {}])


# --- Расширение: множественные поля, секции, счётчики, abstract ---


def test_set_prf_generic_lowercase_normalizer() -> None:
    # keywords сравниваются без учёта регистра/пунктуации.
    p, r, f = set_prf(["Catalysis", "ROS"], ["catalysis", "ros"], str.lower)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_section_headings_flattens_tree() -> None:
    sections = [
        {
            "heading": "Introduction",
            "subsections": [{"heading": "Background", "subsections": []}],
        },
        {"heading": "Methods", "subsections": []},
    ]
    assert section_headings(sections) == ["Introduction", "Background", "Methods"]


def test_evaluate_includes_abstract_when_gold_present() -> None:
    preds = [{"abstract": "We study reactive oxygen species in cells."}]
    golds = [{"abstract": "reactive oxygen species"}]  # вложенность -> fuzzy match

    report = evaluate(preds, golds)

    assert "abstract" in report.fields
    assert report.fields["abstract"].recall == 1.0
    assert report.fields["abstract"].support == 1


def test_evaluate_keywords_and_sections_set_metrics() -> None:
    preds = [{"keywords": ["catalysis", "ros"], "sections": ["Intro", "Methods"]}]
    golds = [{"keywords": ["catalysis"], "sections": ["Intro", "Methods", "Results"]}]

    report = evaluate(preds, golds)

    # keywords: 1 верный + 1 лишний -> P=0.5, R=1.0
    assert report.fields["keywords"].precision == 0.5
    assert report.fields["keywords"].recall == 1.0
    # sections: нашли 2 из 3, без лишних -> P=1.0, R≈0.667
    assert report.fields["sections"].precision == 1.0
    assert round(report.fields["sections"].recall, 3) == 0.667


def test_evaluate_count_fields() -> None:
    # Эталон: 4 фигуры, нашли 3 -> recall=0.75, precision=1.0.
    preds = [{"figures": 3}]
    golds = [{"figures": 4}]

    report = evaluate(preds, golds)

    assert report.fields["figures"].recall == 0.75
    assert report.fields["figures"].precision == 1.0


def test_evaluate_omits_unlabeled_optional_fields() -> None:
    # Если в эталоне нет keywords/sections/abstract — их нет и в отчёте.
    report = evaluate([{"title": "X"}], [{"title": "X"}])
    assert "keywords" not in report.fields
    assert "sections" not in report.fields
    assert "abstract" not in report.fields
    # базовые поля присутствуют всегда
    assert "title" in report.fields
    assert "authors" in report.fields
