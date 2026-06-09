"""Извлечение и обогащение метаданных документа.

Каскад источников (от самого надёжного к запасному):
    1. DOI (regex) -> CrossRef API   -> чистые метаданные, высокий confidence;
    2. локальная LLM по тексту 1-й страницы;
    3. голые эвристики по блокам (размер шрифта, якоря) -- работают всегда (offline).

На вход модуль получает ``list[PageBlock]`` (выход extractor.get_page_blocks),
на выход отдаёт оркестратору готовый объект :class:`parser.schemas.Metadata`.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date
from typing import Any

import httpx
from loguru import logger

from parser.schemas import Metadata, PageBlock

# --- Регулярки и константы ---

#: Канонический DOI: 10.<регистрант>/<суффикс>.
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", re.IGNORECASE)

#: Якорь «это DOI самой статьи»: ссылка/метка непосредственно перед DOI.
_DOI_ANCHOR_RE = re.compile(r"(?:doi\.org/|dx\.doi\.org/|doi:?\s*)$", re.IGNORECASE)

#: Год публикации (4 цифры, разумный диапазон).
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

#: Якоря публикационной даты: год рядом с ними считаем достовернее, чем
#: случайный год из текста/списка литературы.
_YEAR_ANCHORS = (
    "©",
    "(c)",
    "copyright",
    "published",
    "available online",
    "received",
    "accepted",
    "doi.org",
)

#: Хвостовая пунктуация, которую нужно срезать с пойманного DOI.
_DOI_TRAILING = ".,;:)]}>\"'"

#: Хвостовые маркеры аффилиаций. Срезаются с конца строки автора:
#:   - цифры/символы-сноски ("Tilki1,2", "Celikbıcak3") — даже прилипшие к фамилии;
#:   - одиночная строчная буква ("Yao a,*") — только если отделена пробелом/запятой
#:     (lookbehind не даёт «съесть» строчные буквы самой фамилии, напр. "Madonna").
_AFFILIATION_MARKERS = re.compile(
    r"(?:[\s,]*(?:\d+|[*†‡§¶]|(?<=[\s,])[a-z](?![a-z])))+\s*$"
)

#: Разделители авторов в строке byline (", ; · and &").
_AUTHOR_SPLIT = re.compile(r"\s*(?:,|;|·|\band\b|&)\s*", re.IGNORECASE)

#: Признак «в куске есть слово» (буквенная последовательность 2+).
_HAS_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")

#: Якоря, на которых заканчивается зона авторов на титульной странице.
_AUTHOR_STOP_ANCHORS = ("abstract", "keywords", "introduction", "1.")

#: Подстроки служебных блоков (баннеры/ссылки/лицензии) — это не title/authors.
_JUNK_MARKERS = (
    "http://",
    "https://",
    "www.",
    "@",
    "doi.org",
    "doi:",
    "view article online",
    "view journal",
    "view issue",
    "contents lists",
    "sciencedirect",
    "homepage",
    "cite this",
    "creative commons",
    "open access",
    "accepted manuscript",
    "electronic supplementary",
    "supporting information",
    "available online",
    "published on",
    "downloaded on",
    "licence",
    "license",
    "issn",
    "review article",
    "research article",
    "full paper",
)

#: «Журнальные» слова: короткий блок-баннер у верха с таким словом — не заголовок.
_JOURNAL_WORDS = frozenset(
    {
        "journal",
        "review",
        "reviews",
        "letters",
        "communications",
        "proceedings",
        "advances",
        "discussions",
        "transactions",
        "bulletin",
    }
)

#: Частицы фамилий, допустимые в нижнем регистре внутри имени автора.
_NAME_PARTICLES = frozenset(
    {
        "van", "von", "de", "der", "den", "del",
        "dos", "da", "di", "la", "le", "bin", "al",
    }
)

#: Минимальная длина правдоподобного заголовка (короче — зовём LLM).
_MIN_TITLE_LEN = 12

#: Базовый эндпоинт CrossRef REST API.
CROSSREF_BASE_URL = "https://api.crossref.org/works"

#: Таймаут запроса к CrossRef, секунды (API временами отвечает >10с).
CROSSREF_TIMEOUT = 20.0

#: Контактный e-mail для «вежливого пула» CrossRef (стабильнее лимиты).
#: Задаётся проектом; если None — запрос уходит без mailto.
CROSSREF_MAILTO: str | None = None

#: Эндпоинт сервера Ollama (chat API). По умолчанию localhost (offline).
#: Переопределяется переменной окружения OLLAMA_URL — например, в Docker:
#: OLLAMA_URL=http://host.docker.internal:11434/api/chat
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

#: Локальная модель для извлечения метаданных (env OLLAMA_MODEL, см. README).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

#: Таймаут запроса к локальной LLM, секунды.
LLM_TIMEOUT = 60.0

#: Системный промпт: извлечь метаданные строго в JSON.
_LLM_SYSTEM_PROMPT = (
    "Ты извлекаешь метаданные из текста титульной страницы научной статьи. "
    "Верни СТРОГО JSON с ключами: "
    "title (строка), "
    'authors (список строк в формате "Фамилия, Имя"), '
    "abstract (строка или null). "
    "Не добавляй пояснений и markdown."
)


def _is_placeholder_doi(doi: str) -> bool:
    """Отсекает шаблонные DOI-заглушки (напр. RSC ``10.1039/b000000x``)."""
    suffix = doi.split("/", 1)[1] if "/" in doi else doi
    return bool(re.search(r"0{5,}", suffix)) or "xxxx" in suffix


def find_doi(text: str) -> str | None:
    """Находит DOI самой статьи среди всех DOI в тексте.

    В научных статьях, помимо собственного DOI, встречаются DOI цитируемых работ
    (список литературы). Поэтому не берём первый попавшийся, а выбираем по сигналам:
    приоритет — у DOI рядом с якорем (``doi.org/``, ``doi:``); среди них (или среди
    всех, если якорей нет) — самый частый (свой повторяется в колонтитулах/
    строке цитирования), при равенстве — самый ранний по тексту. Шаблоны-заглушки
    отсеиваются.

    Args:
        text: Текст для поиска (приоритетно — титульная страница / front matter).

    Returns:
        DOI в нижнем регистре без хвостовой пунктуации либо ``None``.
    """
    anchored: list[tuple[str, int]] = []
    everything: list[tuple[str, int]] = []
    for match in DOI_RE.finditer(text):
        doi = match.group(0).rstrip(_DOI_TRAILING).lower()
        if _is_placeholder_doi(doi):
            continue
        everything.append((doi, match.start()))
        if _DOI_ANCHOR_RE.search(text[max(0, match.start() - 10) : match.start()]):
            anchored.append((doi, match.start()))

    pool = anchored or everything
    if not pool:
        return None

    counts = Counter(doi for doi, _ in pool)
    # Самый частый DOI; при равенстве — самый ранний по позиции в тексте.
    best_doi = min(pool, key=lambda item: (-counts[item[0]], item[1]))[0]
    logger.debug(f"Выбран DOI: {best_doi} (кандидатов: {len(everything)})")
    return best_doi


def normalize_author(name: str) -> str:
    """Приводит имя автора к формату «Фамилия, Имя».

    Уже нормализованные имена («Yao, Yunjin») возвращаются как есть.
    Из «сырых» имён срезаются маркеры аффилиаций («Yunjin Yao a,*»).

    Args:
        name: Имя автора в произвольном виде.

    Returns:
        Имя в формате «Фамилия, Имя» (или исходное, если разобрать не удалось).
    """
    cleaned = name.strip()
    if not cleaned:
        return ""

    # Сначала срезаем хвостовые маркеры аффилиаций ("a,*", "1,2"): они тоже
    # содержат запятые, поэтому чистим до проверки на формат «Фамилия, Имя».
    cleaned = _AFFILIATION_MARKERS.sub("", cleaned).strip()

    # Уже в формате «Фамилия, Имя» — ничего не трогаем.
    if "," in cleaned:
        return cleaned

    parts = cleaned.split()
    if len(parts) < 2:
        return cleaned

    surname = parts[-1]
    given = " ".join(parts[:-1])
    return f"{surname}, {given}"


def _text_blocks(blocks: list[PageBlock], page: int) -> list[PageBlock]:
    """Текстовые блоки указанной страницы (без картинок и пустых, 1-индексация)."""
    return [
        b for b in blocks if b.page_number == page and b.block_type == "text" and b.text
    ]


def _is_junk_text(text: str) -> bool:
    """Похоже на служебный блок (баннер/ссылка/лицензия), а не title/authors."""
    low = text.lower()
    return any(marker in low for marker in _JUNK_MARKERS)


def _looks_like_masthead(block: PageBlock) -> bool:
    """Похоже на баннер журнала/метку рубрики у верха страницы, а не заголовок."""
    text = (block.text or "").strip()
    if block.bbox.top > 150:
        return False
    words = text.split()
    # Короткий блок у самого верха — обычно аббревиатура/название журнала.
    if len(words) <= 4 and len(text) < 40:
        return True
    # Короткий блок с «журнальным» словом (Journal/Review/Letters/...).
    lowered = {w.strip(".,:").lower() for w in words}
    return len(words) <= 6 and bool(lowered & _JOURNAL_WORDS)


def _looks_like_person(name: str) -> bool:
    """Похоже на имя человека: заглавные токены, без служебных слов и мусора."""
    if _is_junk_text(name):
        return False
    tokens = [t for t in name.replace(",", " ").split() if t]
    if not tokens or len(tokens) > 5:
        return False
    capitalized = 0
    for token in tokens:
        core = token.strip(".-")
        if not core or core.lower() in _NAME_PARTICLES:
            continue
        if core[:1].isupper():
            capitalized += 1
        else:
            # Строчное слово, не являющееся частицей фамилии — это не имя.
            return False
    return capitalized >= 1


def _pick_title_block(first_page: list[PageBlock]) -> PageBlock:
    """Выбирает блок-заголовок: крупнейший шрифт среди не-мусорных, не-баннеров."""
    candidates = [
        b
        for b in first_page
        if b.text and not _is_junk_text(b.text) and not _looks_like_masthead(b)
    ]
    pool = candidates or first_page  # фоллбэк: вернуть хоть что-то
    return max(pool, key=lambda b: (b.font_size or 0.0, b.is_bold, -b.bbox.top))


def guess_title(blocks: list[PageBlock]) -> str:
    """Эвристика заголовка: крупнейший блок стр. 1 без баннеров/служебки.

    Отсекаются блоки-«мусор» (URL, «View Article Online», лицензии) и баннеры
    журнала (короткий блок у верха или с «журнальным» словом).

    Args:
        blocks: Все блоки документа (выход extractor.get_page_blocks).

    Returns:
        Текст заголовка. Пустая строка, если блоков нет (схема требует ``str``).
    """
    first_page = _text_blocks(blocks, 1)
    if not first_page:
        return ""
    return (_pick_title_block(first_page).text or "").strip()


def guess_abstract(blocks: list[PageBlock]) -> str | None:
    """Эвристика аннотации: блок с якорем «Abstract» на первых страницах.

    Args:
        blocks: Все блоки документа.

    Returns:
        Текст аннотации либо ``None``, если якорь не найден.
    """
    candidates = _text_blocks(blocks, 1) + _text_blocks(blocks, 2)
    for i, block in enumerate(candidates):
        text = (block.text or "").strip()
        if text.lower().startswith("abstract"):
            # "Abstract: текст..." — берём остаток той же строки.
            remainder = re.sub(
                r"^abstract[\s:.\-—]*", "", text, flags=re.IGNORECASE
            ).strip()
            if remainder:
                return remainder
            # Заголовок "Abstract" отдельной строкой — берём следующий блок.
            if i + 1 < len(candidates):
                return candidates[i + 1].text
    return None


def _parse_author_line(text: str) -> list[str]:
    """Разбивает строку byline на отдельных авторов в формате «Фамилия, Имя».

    Куски-маркеры аффилиаций ("a", "*", "1") отсеиваются как «без слова».
    """
    authors: list[str] = []
    seen: set[str] = set()
    for piece in _AUTHOR_SPLIT.split(text):
        piece = piece.strip()
        if not piece or _HAS_WORD.search(piece) is None:
            continue
        name = normalize_author(piece)
        if name and _HAS_WORD.search(name) is not None and name not in seen:
            seen.add(name)
            authors.append(name)
    return authors


def guess_authors(blocks: list[PageBlock]) -> list[str]:
    """Эвристика авторов: первый «именной» блок под заголовком на стр. 1.

    Авторы обычно расположены сразу под названием и до аннотации/аффилиаций.

    Args:
        blocks: Все блоки документа.

    Returns:
        Список авторов «Фамилия, Имя» либо ``[]``, если разобрать не удалось.
    """
    first_page = _text_blocks(blocks, 1)
    if not first_page:
        return []

    title_block = _pick_title_block(first_page)
    below = sorted(
        (b for b in first_page if b.bbox.top > title_block.bbox.top),
        key=lambda b: b.bbox.top,
    )
    for block in below:
        text = (block.text or "").strip()
        if text.lower().startswith(_AUTHOR_STOP_ANCHORS):
            break
        if _is_junk_text(text):
            continue
        names = [n for n in _parse_author_line(text) if _looks_like_person(n)]
        if names:
            return names
    return []


def guess_year(text: str) -> int | None:
    """Эвристика года публикации.

    Приоритет — год рядом с публикационными якорями (©, published, received,
    accepted, doi.org); иначе самый поздний правдоподобный год (ссылки в статье
    не бывают новее года публикации). Неправдоподобные годы отсекаются.

    Args:
        text: Текст для поиска (приоритетно — титульная страница / front matter).

    Returns:
        Год публикации либо ``None``.
    """
    max_year = date.today().year + 1
    low = text.lower()
    anchored: list[int] = []
    plausible: list[int] = []
    for match in YEAR_RE.finditer(text):
        year = int(match.group(0))
        if not 1900 <= year <= max_year:
            continue
        plausible.append(year)
        window = low[max(0, match.start() - 40) : match.end() + 40]
        if any(anchor in window for anchor in _YEAR_ANCHORS):
            anchored.append(year)
    if anchored:
        return max(anchored)
    if plausible:
        return max(plausible)
    return None


# --- Заглушки внешнего обогащения (CrossRef / LLM) ---
# Реализация — следующий этап. Сейчас возвращают None, чтобы каскад
# корректно проваливался в оффлайн-эвристики и проходил mypy --strict.


def query_crossref(doi: str) -> dict[str, Any] | None:
    """Запрашивает метаданные по DOI в CrossRef API.

    Опциональна, отключается флагами ``--offline`` / ``--no-crossref``.
    Любая сетевая ошибка/таймаут/битый ответ => ``None`` (каскад уходит в эвристики).

    Args:
        doi: Канонический DOI.

    Returns:
        Объект ``message`` из ответа CrossRef либо ``None``.
    """
    user_agent = (
        "pdf-to-json-parser/0.1 (https://github.com/mihaillogger/pdf-to-json-parser)"
    )
    if CROSSREF_MAILTO:
        user_agent += f" mailto:{CROSSREF_MAILTO}"

    try:
        response = httpx.get(
            f"{CROSSREF_BASE_URL}/{doi}",
            headers={"User-Agent": user_agent},
            timeout=CROSSREF_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(f"CrossRef недоступен для DOI {doi}: {exc}")
        return None

    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict):
        logger.warning(f"CrossRef вернул неожиданный ответ для DOI {doi}")
        return None
    logger.debug(f"CrossRef отдал метаданные по DOI {doi}")
    return message


def query_llm(page_text: str, *, model: str | None = None) -> dict[str, Any] | None:
    """Извлекает метаданные локальной LLM (Ollama) из текста титульной страницы.

    Работает оффлайн: запрос идёт на localhost к локальному серверу Ollama.
    Любая ошибка (сервер не поднят / таймаут / невалидный JSON) => ``None``.

    Args:
        page_text: Текст первой страницы.
        model: Имя модели Ollama (по умолчанию :data:`OLLAMA_MODEL`).

    Returns:
        Словарь с ключами title/authors/abstract либо ``None``.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": page_text},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(f"Локальная LLM недоступна: {exc}")
        return None

    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        logger.warning("LLM вернула неожиданный ответ")
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning(f"LLM вернула невалидный JSON: {exc}")
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_metadata(
    blocks: list[PageBlock],
    raw_text: str,
    *,
    use_crossref: bool = True,
    use_llm: bool = True,
    offline: bool = False,
) -> Metadata:
    """Собирает метаданные документа по каскаду источников.

    Args:
        blocks: Стандартизированные блоки страниц (extractor.get_page_blocks).
        raw_text: Полный текст документа (для поиска DOI/года).
        use_crossref: Разрешить обращение к CrossRef API.
        use_llm: Разрешить локальную LLM как фоллбэк.
        offline: Полностью оффлайн (CrossRef принудительно выключен).

    Returns:
        Объект :class:`parser.schemas.Metadata` со всеми обязательными полями.
    """
    # offline имеет приоритет: в оффлайне сеть выключена при любом use_crossref.
    crossref_enabled = use_crossref and not offline
    logger.debug(
        f"Режим метаданных: crossref={crossref_enabled}, "
        f"llm={use_llm}, offline={offline}"
    )

    doi = find_doi(raw_text)

    # 1. CrossRef по DOI — самый чистый источник.
    if doi is not None and crossref_enabled:
        crossref = query_crossref(doi)
        if crossref is not None:
            return _build_from_crossref(doi, crossref)

    # 2. Оффлайн-эвристики по блокам (всегда доступны).
    title = guess_title(blocks)
    abstract = guess_abstract(blocks)
    authors = guess_authors(blocks)
    year = guess_year(raw_text)

    source = "pdf"
    confidence = 0.4 if title else 0.2

    # 3. LLM-фоллбэк, если эвристики дали мало ИЛИ подозрительный результат:
    #    нет авторов / нет аннотации / заголовок пустой или слишком короткий
    #    (например, остался баннер журнала, который не отсеяли фильтры).
    weak_title = not title or len(title) < _MIN_TITLE_LEN
    if use_llm and (weak_title or abstract is None or not authors):
        first_page_text = "\n".join(b.text or "" for b in _text_blocks(blocks, 1))
        llm = query_llm(first_page_text)
        if llm is not None:
            # LLM только ЗАПОЛНЯЕТ пробелы, не перезаписывает удачные эвристики
            # (модель меньше и склонна привирать авторов/заголовок).
            used = False
            if weak_title and llm.get("title"):
                title = str(llm["title"]).strip()
                used = True
            if abstract is None and llm.get("abstract"):
                abstract = str(llm["abstract"])
                used = True
            if not authors and isinstance(llm.get("authors"), list):
                llm_authors = [
                    normalize_author(str(a)) for a in llm["authors"] if str(a).strip()
                ]
                authors = [a for a in llm_authors if _looks_like_person(a)]
                used = used or bool(authors)
            if used:
                source = "hybrid"
                confidence = 0.6

    return Metadata(
        title=title,
        title_en=None,
        authors=authors,
        abstract=abstract,
        keywords=[],
        doi=doi,
        journal=None,
        year=year,
        metadata_source=source,
        metadata_confidence=confidence,
        normative=None,
    )


def _first(value: Any) -> str | None:
    """Берёт первый непустой элемент списка строк (формат полей CrossRef)."""
    if isinstance(value, list) and value:
        head = value[0]
        return head if isinstance(head, str) and head.strip() else None
    return None


def _crossref_authors(message: dict[str, Any]) -> list[str]:
    """Собирает авторов в формате «Фамилия, Имя» из поля ``author`` CrossRef."""
    authors: list[str] = []
    for entry in message.get("author", []):
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family") or "").strip()
        given = str(entry.get("given") or "").strip()
        if family and given:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)
        elif entry.get("name"):
            authors.append(normalize_author(str(entry["name"])))
    return authors


def _crossref_year(message: dict[str, Any]) -> int | None:
    """Извлекает год публикации из date-parts (issued/published/...)."""
    for key in ("issued", "published", "published-print", "published-online"):
        date_parts = message.get(key, {})
        if not isinstance(date_parts, dict):
            continue
        parts = date_parts.get("date-parts")
        if not (isinstance(parts, list) and parts):
            continue
        head = parts[0]
        if isinstance(head, list) and head and isinstance(head[0], int):
            return head[0]
    return None


def _crossref_abstract(message: dict[str, Any]) -> str | None:
    """Достаёт abstract, очищая JATS/XML-теги (<jats:p> и т.п.)."""
    raw = message.get("abstract")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = re.sub(r"<[^>]+>", "", raw).strip()
    return text or None


def _build_from_crossref(doi: str, message: dict[str, Any]) -> Metadata:
    """Собирает Metadata из объекта ``message`` ответа CrossRef.

    Args:
        doi: Канонический DOI (используется как итоговое значение поля).
        message: Объект ``message`` из ответа CrossRef.

    Returns:
        Заполненный объект :class:`Metadata` с источником ``crossref``.
    """
    subjects = message.get("subject")
    keywords = (
        [s for s in subjects if isinstance(s, str)]
        if isinstance(subjects, list)
        else []
    )
    return Metadata(
        title=_first(message.get("title")) or "",
        title_en=None,
        authors=_crossref_authors(message),
        abstract=_crossref_abstract(message),
        keywords=keywords,
        doi=doi,
        journal=_first(message.get("container-title")),
        year=_crossref_year(message),
        metadata_source="crossref",
        metadata_confidence=0.9,
        normative=None,
    )
