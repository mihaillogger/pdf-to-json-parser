"""Тесты модуля извлечения метаданных (parser.metadata)."""

from __future__ import annotations

from typing import Any

import httpx

import parser.metadata as metadata
from parser.metadata import (
    extract_metadata,
    find_doi,
    guess_abstract,
    guess_authors,
    guess_title,
    guess_year,
    normalize_author,
)
from parser.schemas import BBox, PageBlock


def _block(
    text: str,
    font_size: float,
    top: float,
    page: int = 1,
    *,
    is_bold: bool = False,
    block_type: str = "text",
) -> PageBlock:
    """Удобный конструктор блока в формате extractor.get_page_blocks()."""
    return PageBlock(
        text=text,
        font_size=font_size,
        bbox=BBox(left=50.0, top=top, right=550.0, bottom=top + 20.0),
        page_number=page,
        block_type=block_type,
        is_bold=is_bold,
    )


# --- find_doi ---


def test_find_doi_canonical() -> None:
    text = "Available online. https://doi.org/10.1016/j.ces.2025.121219 Elsevier"
    assert find_doi(text) == "10.1016/j.ces.2025.121219"


def test_find_doi_strips_trailing_punctuation() -> None:
    assert find_doi("see DOI 10.1021/acs.jpcc.1c00123.") == "10.1021/acs.jpcc.1c00123"


def test_find_doi_lowercased() -> None:
    assert find_doi("DOI: 10.1016/J.CES.2025.121219") == "10.1016/j.ces.2025.121219"


def test_find_doi_absent() -> None:
    assert find_doi("В этом тексте нет идентификатора публикации") is None


def test_find_doi_prefers_own_over_cited() -> None:
    # Свой DOI в шапке + повтор в футере; чужие — по разу в списке литературы.
    text = (
        "Article. https://doi.org/10.1016/j.own.2025.001\n"
        "Body text...\n"
        "References\n"
        "[1] Smith et al. https://doi.org/10.1111/cited.2010.999\n"
        "[2] Jones et al. https://doi.org/10.2222/cited.2011.888\n"
        "Footer https://doi.org/10.1016/j.own.2025.001\n"
    )
    assert find_doi(text) == "10.1016/j.own.2025.001"


def test_find_doi_prefers_anchored() -> None:
    # Без повторов: DOI у якоря doi.org предпочтительнее голого в теле.
    text = "ref 10.9999/loose.body.ref ... official doi.org/10.1016/j.real.2024.42"
    assert find_doi(text) == "10.1016/j.real.2024.42"


def test_find_doi_skips_placeholder() -> None:
    # Шаблон-заглушка RSC не должен выбираться, если есть настоящий DOI.
    text = "DOI: 10.1039/b000000x ... real one https://doi.org/10.1039/d2cs00172a"
    assert find_doi("10.1039/b000000x") is None
    assert find_doi(text) == "10.1039/d2cs00172a"


# --- normalize_author ---


def test_normalize_author_plain() -> None:
    assert normalize_author("Yunjin Yao") == "Yao, Yunjin"


def test_normalize_author_already_formatted() -> None:
    assert normalize_author("Yao, Yunjin") == "Yao, Yunjin"


def test_normalize_author_strips_affiliation_markers() -> None:
    assert normalize_author("Yunjin Yao a,*") == "Yao, Yunjin"


def test_normalize_author_single_token() -> None:
    assert normalize_author("Madonna") == "Madonna"


def test_normalize_author_attached_affiliation_digits() -> None:
    # Реальный кейс: цифры аффилиаций прилипают к фамилии без разделителя.
    assert normalize_author("Serhad Tilki1,2") == "Tilki, Serhad"
    assert normalize_author("Omur Celikbıcak3") == "Celikbıcak, Omur"
    assert normalize_author("Mehmet Yakup Arica1,5") == "Arica, Mehmet Yakup"


# --- guess_title ---


def test_guess_title_picks_largest_font_on_page_1() -> None:
    blocks = [
        _block("Some journal header", 8.0, top=10.0),
        _block("Manganese-iron supported on porous iron foam", 18.0, top=40.0),
        _block("Yunjin Yao, Yating Liu", 10.0, top=70.0),
    ]
    assert guess_title(blocks) == "Manganese-iron supported on porous iron foam"


def test_guess_title_ignores_image_blocks() -> None:
    blocks = [
        PageBlock(
            text=None,
            font_size=None,
            bbox=BBox(left=0.0, top=0.0, right=600.0, bottom=400.0),
            page_number=1,
            block_type="image",
            is_bold=False,
        ),
        _block("Real Title Here", 16.0, top=40.0),
    ]
    assert guess_title(blocks) == "Real Title Here"


def test_guess_title_empty_when_no_blocks() -> None:
    assert guess_title([]) == ""


# --- guess_abstract ---


def test_guess_abstract_inline() -> None:
    blocks = [_block("Abstract: The development of porous catalysts.", 10.0, top=100.0)]
    assert guess_abstract(blocks) == "The development of porous catalysts."


def test_guess_abstract_next_block() -> None:
    blocks = [
        _block("Abstract", 11.0, top=100.0),
        _block("The development of highly active catalysts.", 10.0, top=120.0),
    ]
    assert guess_abstract(blocks) == "The development of highly active catalysts."


def test_guess_abstract_absent() -> None:
    assert guess_abstract([_block("Introduction", 10.0, top=100.0)]) is None


# --- guess_authors ---


def test_guess_authors_byline_with_affiliations() -> None:
    blocks = [
        _block("Catalytic oxidation over iron foam", 18.0, top=40.0),
        _block("Yunjin Yao a,* , Yating Liu a , Zhenshan Ma b", 10.0, top=70.0),
        _block("Abstract: ...", 10.0, top=100.0),
    ]
    assert guess_authors(blocks) == ["Yao, Yunjin", "Liu, Yating", "Ma, Zhenshan"]


def test_guess_authors_with_and_separator() -> None:
    blocks = [
        _block("A Great Paper", 18.0, top=40.0),
        _block("John Smith and Jane Doe", 10.0, top=70.0),
    ]
    assert guess_authors(blocks) == ["Smith, John", "Doe, Jane"]


def test_guess_authors_stops_at_abstract() -> None:
    blocks = [
        _block("A Great Paper", 18.0, top=40.0),
        _block("Abstract", 11.0, top=70.0),
        _block("Some text that mentions a Name", 10.0, top=90.0),
    ]
    assert guess_authors(blocks) == []


def test_guess_authors_empty_when_no_blocks() -> None:
    assert guess_authors([]) == []


# --- guess_year ---


def test_guess_year_prefers_later_anchored() -> None:
    # received 2024 и accepted 2025 — оба у якорей, берём поздний (год публикации).
    assert guess_year("Received 2024; accepted 2025") == 2025


def test_guess_year_anchored_over_random() -> None:
    # Год у © приоритетнее случайного года из тела статьи.
    text = "...as shown in 1998 studies... © 2021 Elsevier B.V."
    assert guess_year(text) == 2021


def test_guess_year_fallback_to_latest_plausible() -> None:
    # Без якорей — самый поздний правдоподобный (ссылки не новее публикации).
    assert guess_year("refs: 2011, 2019, 2015 ... no anchors") == 2019


def test_guess_year_ignores_implausible() -> None:
    assert guess_year("code 3099 and id 1500") is None


def test_guess_year_absent() -> None:
    assert guess_year("no dates here") is None


# --- extract_metadata (оффлайн-каскад) ---


def test_extract_metadata_offline_heuristics() -> None:
    blocks = [
        _block("Catalytic oxidation over iron foam", 18.0, top=40.0),
        _block("Yunjin Yao a , Yating Liu b", 10.0, top=60.0),
        _block("Abstract: A study of catalysts.", 10.0, top=80.0),
    ]
    raw_text = "Catalytic oxidation... DOI 10.1016/j.ces.2025.121219 ... 2025"

    meta = extract_metadata(blocks, raw_text, offline=True)

    assert meta.title == "Catalytic oxidation over iron foam"
    assert meta.abstract == "A study of catalysts."
    assert meta.doi == "10.1016/j.ces.2025.121219"
    assert meta.year == 2025
    assert meta.metadata_source == "pdf"
    assert 0.0 <= meta.metadata_confidence <= 1.0
    assert meta.authors == ["Yao, Yunjin", "Liu, Yating"]


# --- Логика флагов режимов (online / offline+llm / offline) ---


def _record_calls(monkeypatch: object) -> dict[str, int]:
    """Подменяет query_crossref/query_llm счётчиками вызовов."""
    calls = {"crossref": 0, "llm": 0}

    def fake_crossref(doi: str) -> None:
        calls["crossref"] += 1
        return None

    def fake_llm(page_text: str, *, model: str | None = None) -> None:
        calls["llm"] += 1
        return None

    monkeypatch.setattr(metadata, "query_crossref", fake_crossref)  # type: ignore[attr-defined]
    monkeypatch.setattr(metadata, "query_llm", fake_llm)  # type: ignore[attr-defined]
    return calls


def test_offline_disables_crossref(monkeypatch: object) -> None:
    # offline имеет приоритет: CrossRef не вызывается даже при use_crossref=True.
    calls = _record_calls(monkeypatch)
    blocks = [_block("Some Title", 18.0, top=40.0)]
    raw_text = "DOI 10.1016/j.ces.2025.121219"

    extract_metadata(blocks, raw_text, use_crossref=True, offline=True)

    assert calls["crossref"] == 0


def test_online_calls_crossref_when_doi_present(monkeypatch: object) -> None:
    calls = _record_calls(monkeypatch)
    blocks = [_block("Some Title", 18.0, top=40.0)]
    raw_text = "DOI 10.1016/j.ces.2025.121219"

    extract_metadata(blocks, raw_text, offline=False)

    assert calls["crossref"] == 1


def test_no_llm_flag_skips_llm(monkeypatch: object) -> None:
    # use_llm=False -> LLM не вызывается, даже когда эвристики дали мало.
    calls = _record_calls(monkeypatch)
    blocks = [_block("x", 18.0, top=40.0)]  # нет abstract -> повод для LLM-фоллбэка

    extract_metadata(blocks, "no doi here", offline=True, use_llm=False)

    assert calls["llm"] == 0


def test_offline_llm_invokes_llm_fallback(monkeypatch: object) -> None:
    # Режим «оффлайн + LLM»: нет abstract -> LLM-фоллбэк вызывается.
    calls = _record_calls(monkeypatch)
    blocks = [_block("Only a title block", 18.0, top=40.0)]

    extract_metadata(blocks, "no doi here", offline=True, use_llm=True)

    assert calls["crossref"] == 0
    assert calls["llm"] == 1


# --- CrossRef: парсинг ответа и сеть ---

SAMPLE_CROSSREF_MESSAGE: dict[str, Any] = {
    "title": ["Manganese-iron supported on porous iron foam"],
    "author": [
        {"given": "Yunjin", "family": "Yao", "sequence": "first"},
        {"given": "Yating", "family": "Liu"},
    ],
    "container-title": ["Chemical Engineering Science"],
    "issued": {"date-parts": [[2025, 3, 1]]},
    "abstract": "<jats:p>The development of highly active catalysts.</jats:p>",
    "subject": ["General Chemical Engineering"],
    "DOI": "10.1016/j.ces.2025.121219",
}


def test_build_from_crossref_parses_fields() -> None:
    meta = metadata._build_from_crossref(
        "10.1016/j.ces.2025.121219", SAMPLE_CROSSREF_MESSAGE
    )
    assert meta.title == "Manganese-iron supported on porous iron foam"
    assert meta.authors == ["Yao, Yunjin", "Liu, Yating"]
    assert meta.journal == "Chemical Engineering Science"
    assert meta.year == 2025
    assert meta.abstract == "The development of highly active catalysts."
    assert meta.keywords == ["General Chemical Engineering"]
    assert meta.metadata_source == "crossref"
    assert meta.metadata_confidence == 0.9


class _FakeResponse:
    """Минимальная замена httpx.Response для тестов."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_query_crossref_success(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"status": "ok", "message": SAMPLE_CROSSREF_MESSAGE})

    monkeypatch.setattr(httpx, "get", fake_get)
    message = metadata.query_crossref("10.1016/j.ces.2025.121219")

    assert message is not None
    assert message["container-title"] == ["Chemical Engineering Science"]


def test_query_crossref_network_error_returns_none(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert metadata.query_crossref("10.1016/j.ces.2025.121219") is None


def test_extract_metadata_online_uses_crossref(monkeypatch: Any) -> None:
    monkeypatch.setattr(metadata, "query_crossref", lambda doi: SAMPLE_CROSSREF_MESSAGE)
    blocks = [_block("heuristic title to be overridden", 18.0, top=40.0)]
    raw_text = "see DOI 10.1016/j.ces.2025.121219"

    meta = extract_metadata(blocks, raw_text, offline=False)

    assert meta.metadata_source == "crossref"
    assert meta.metadata_confidence == 0.9
    assert meta.title == "Manganese-iron supported on porous iron foam"
    assert meta.authors == ["Yao, Yunjin", "Liu, Yating"]


# --- LLM (Ollama): парсинг ответа и сеть ---


def _fake_ollama(content: str) -> "_FakeResponse":
    """Ответ Ollama chat API: message.content — это JSON-строка от модели."""
    return _FakeResponse({"message": {"role": "assistant", "content": content}})


def test_query_llm_success(monkeypatch: Any) -> None:
    reply = '{"title": "A Paper", "authors": ["Smith, John"], "abstract": "Text."}'

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _fake_ollama(reply)

    monkeypatch.setattr(httpx, "post", fake_post)
    result = metadata.query_llm("some page text")

    assert result == {
        "title": "A Paper",
        "authors": ["Smith, John"],
        "abstract": "Text.",
    }


def test_query_llm_network_error_returns_none(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        raise httpx.ConnectError("ollama not running")

    monkeypatch.setattr(httpx, "post", fake_post)
    assert metadata.query_llm("text") is None


def test_query_llm_invalid_json_returns_none(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _fake_ollama("это не json, а просто текст")

    monkeypatch.setattr(httpx, "post", fake_post)
    assert metadata.query_llm("text") is None


def test_extract_metadata_offline_llm_fallback(monkeypatch: Any) -> None:
    # Эвристики не нашли abstract -> LLM-фоллбэк отдаёт данные, source=hybrid.
    monkeypatch.setattr(
        metadata,
        "query_llm",
        lambda page_text, **kw: {
            "title": "LLM Title",
            "authors": ["Doe, Jane"],
            "abstract": "LLM abstract.",
        },
    )
    blocks = [_block("weak title", 12.0, top=40.0)]

    meta = extract_metadata(blocks, "no doi", offline=True, use_llm=True)

    assert meta.metadata_source == "hybrid"
    assert meta.title == "LLM Title"
    assert meta.authors == ["Doe, Jane"]
    assert meta.abstract == "LLM abstract."
