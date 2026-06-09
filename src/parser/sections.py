"""
Модуль для сборки иерархического дерева секций научного документа.

Использует конечный автомат на основе стека для преобразования плоского
массива текстовых блоков (PageBlock) в рекурсивную структуру (Section).
Определение заголовков базируется на эвристиках: анализе размера шрифта,
жирности, регулярных выражениях для нумерации и словаре стандартной
онтологии научных статей.
"""

import re
from collections import Counter

from parser.schemas import PageBlock, Section

STANDARD_SECTION_ONTOLOGY = {
    "introduction",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "experimental section",
    "materials and methods",
    "acknowledgements",
    "references",
    "methods",
    "abstract",
}

SKIP_HEADER_KEYWORDS = {
    "keywords:",
    "graphical abstract",
    "highlights",
    "supporting information",
    "full paper",
    "table of contents",
    "published online",
    "doi:",
    "corrigendum",
    "received ",
    "accepted ",
    "revised ",
    "available online",
}


def _get_base_font_size(blocks: list[PageBlock]) -> float:
    """Определяет базовый размер шрифта основного текста документа.

    Args:
        blocks: Список извлеченных блоков страницы.

    Returns:
        float: Наиболее часто встречающийся размер шрифта (мода).
    """
    fonts = [
        b.font_size
        for b in blocks
        if b.block_type == "text" and b.font_size is not None
    ]
    if not fonts:
        return 12.0

    counter = Counter(fonts)
    return counter.most_common(1)[0][0]


def _analyze_heading(
    text: str,
    font_size: float,
    base_font: float,
    is_bold: bool,
    title_found: bool,
    in_references: bool,
) -> tuple[bool, int, str | None]:
    """Анализирует текстовый блок на принадлежность к заголовку секции.

    Опирается на эвристики размера шрифта, жирности, регулярные выражения
    для нумерации и словарь стандартных названий секций.

    Args:
        text: Содержимое текстового блока.
        font_size: Максимальный размер шрифта в блоке.
        base_font: Базовый размер шрифта документа.
        is_bold: Флаг наличия жирного начертания.
        title_found: Флаг, указывающий, был ли уже найден заголовок H1.
        in_references: Флаг нахождения парсера внутри списка литературы.

    Returns:
        tuple: (Является ли заголовком, Уровень вложенности, Номер секции или None).
    """
    text = text.strip()
    text_lower = text.lower()

    # Если мы провалились в References, ловим только подзаголовки
    if in_references:
        if not re.match(r"^(appendix|supplementary|section\s+s|s\d+)", text_lower):
            return False, 0, None

    # Отсекаем подписи к рисункам и таблицам, чтобы они не ломали дерево
    if re.match(r"^(figure|fig\.|table|scheme)\s*\d+", text_lower):
        return False, 0, None

    # Пропуск для известных заголовков
    if text_lower in STANDARD_SECTION_ONTOLOGY:
        return True, 2, None

    if len(text) < 4 or len(text) > 300:
        return False, 0, None

    # Игнорируем блоки, состоящие только из цифр и единиц измерения (координаты, физ. величины)
    if re.fullmatch(
        r"[\d\s\.\,\-]+(?:mM|cm|mm|mV|mA|h|min|s|kΩ|µm)?", text, re.IGNORECASE
    ):
        return False, 0, None

    # Регулярка для отлова нумерованных заголовков
    match = re.match(
        r"^((?:Section\s+)?S\d+|(?:\d+\.)+\d*|\d+\.?|[IVX]+\.)[\s\-]+(.+)",
        text,
        re.IGNORECASE,
    )
    if match:
        number_str = (
            match.group(1).strip(" .").replace("Section ", "").replace("section ", "")
        )
        rest_text = match.group(2)

        if (
            len(text) < 150
            and "\n" not in text
            and len(rest_text) > 2
            and any(c.isalpha() for c in rest_text)
        ):
            if text.endswith(".") or (rest_text and rest_text[0].islower()):
                return False, 0, None

            # В заголовках редко бывает много запятых (защита от списков авторов/аффилиаций)
            if text.count(",") >= 2:
                return False, 0, None

            # Защита от ложных срабатываний на инициалах авторов (например, "A. Smith")
            if not re.match(r"^[A-Z]\.\s+[A-Z]", rest_text):
                level = len(number_str.split(".")) + 1
                return True, level, number_str

    # Эвристика H1: шрифт значительно больше базового (обычно название статьи)
    if font_size > base_font + 2.5:
        if not title_found:
            return True, 1, None

    # Эвристика H2+: шрифт жирный и не меньше базового
    elif is_bold and font_size >= base_font - 0.5:
        if not text.endswith(".") and len(text) < 100 and "\n" not in text:

            # Срезаем стартовые спецсимволы, чтобы проверить первую букву
            clean_start = re.sub(r"^[^a-zA-Z]+", "", text)
            is_lower_start = bool(clean_start and clean_start[0].islower())

            # Если много пунктуации - это скорее всего мусор, а не заголовок
            has_too_much_punct = (text.count(",") >= 2) or (";" in text)

            if not is_lower_start and not has_too_much_punct:
                if len(text.split()) > 1 or text_lower in STANDARD_SECTION_ONTOLOGY:
                    if not text_lower.startswith("abstract:"):
                        return True, 2, None

    return False, 0, None


def build_section_tree(blocks: list[PageBlock]) -> list[Section]:
    """Преобразует плоский массив текстовых блоков в иерархическое дерево секций.

    Использует алгоритм на основе стека для отслеживания текущего уровня вложенности
    и правильной привязки подсекций к родительским узлам.

    Args:
        blocks: Отсортированный массив объектов PageBlock.

    Returns:
        list[Section]: Список корневых секций документа.
    """
    if not blocks:
        return []

    base_font = _get_base_font_size(blocks)
    root_sections: list[Section] = []

    # Стек для отслеживания текущей ветки дерева
    stack: list[Section] = []
    # Трекер размеров шрифта для динамического вычисления уровней (H2, H3 и т.д.)
    level_fonts: dict[int, float] = {}

    current_section = None
    title_found = False
    in_references = False

    for block in blocks:
        if block.block_type == "image":
            continue

        text = block.text.strip() if block.text else ""
        if not text:
            continue

        # Разделяем слипшиеся заголовки (например, "Abstract\nHere is the text")
        lines = text.split("\n", 1)
        if len(lines) == 2 and lines[0].strip().lower() in STANDARD_SECTION_ONTOLOGY:
            chunks = [
                (lines[0].strip(), block.font_size, block.is_bold),
                (lines[1].strip(), block.font_size, False),
            ]
        else:
            chunks = [(text, block.font_size, block.is_bold)]

        for chunk_text, chunk_font, chunk_bold in chunks:
            if any(chunk_text.lower().startswith(kw) for kw in SKIP_HEADER_KEYWORDS):
                continue

            chunk_font = chunk_font or base_font
            is_heading, level, number = _analyze_heading(
                chunk_text,
                chunk_font,
                base_font,
                chunk_bold,
                title_found,
                in_references,
            )

            if is_heading:
                if level == 1:
                    title_found = True
                if chunk_text.lower() == "references":
                    in_references = True

                # Динамическое вычисление уровня вложенности на основе падения/роста размера шрифта
                if number is None and stack and level >= 2:
                    current_parent_level = stack[-1].level
                    current_parent_font = level_fonts.get(current_parent_level, 0.0)

                    if chunk_font < current_parent_font - 0.5:
                        level = current_parent_level + 1
                    elif chunk_font > current_parent_font + 0.5:
                        level = max(2, current_parent_level - 1)
                    else:
                        level = current_parent_level

                level_fonts[level] = chunk_font

                new_section = Section(
                    heading=chunk_text,
                    level=level,
                    content="",
                    subsections=[],
                    number=number,
                    status=None,
                    status_effective_from=None,
                )

                # Выкидываем из стека все секции, чей уровень больше или равен текущему
                while stack and stack[-1].level >= level:
                    stack.pop()

                if not stack:
                    root_sections.append(new_section)
                else:
                    stack[-1].subsections.append(new_section)

                stack.append(new_section)
                current_section = new_section
            else:
                # Пропускаем мусор с титульной страницы (аффилиации, почты),
                # пока не встретим первый валидный заголовок.
                if current_section:
                    if current_section.content:
                        current_section.content += f"\n\n{chunk_text}"
                    else:
                        current_section.content = chunk_text

    return root_sections


def extract_acknowledgments(sections: list[Section]) -> str | None:
    """Рекурсивно ищет, извлекает и удаляет раздел благодарностей из дерева.

    Мутирует переданный список `sections`, удаляя найденный узел, чтобы
    избежать дублирования текста в итоговом JSON.

    Args:
        sections: Список секций (текущий уровень дерева).

    Returns:
        str | None: Текст раздела благодарностей или None.
    """
    target_keywords = {"acknowledgements", "acknowledgments", "funding"}

    for i, section in enumerate(sections):
        if section.heading.strip().lower() in target_keywords:
            ack_text = section.content.strip()
            sections.pop(i)
            return ack_text if ack_text else None

        if section.subsections:
            ack_text = extract_acknowledgments(section.subsections)
            if ack_text:
                return ack_text

    return None
