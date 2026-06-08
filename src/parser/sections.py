import re
from collections import Counter
from typing import List, Optional, Tuple

from parser.schemas import PageBlock, Section

# Фоллбэк для сломанных PDF
# Используется как запасной механизм определения секций, если
# PDF-документ потерял метаданные жирности шрифта.
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

# Ключевые слова для отсечения мусорных колонтитулов и врезок
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


def _get_base_font_size(blocks: List[PageBlock]) -> float:
    """
    Вычисляет базовый размер шрифта документа (обычный текст).

    Args:
        blocks: Список всех блоков документа.

    Returns:
        Наиболее часто встречающийся размер шрифта.
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
) -> Tuple[bool, int, Optional[str]]:
    """
    Анализирует текстовый блок и определяет, является ли он заголовком.
    Опирается на размер шрифта, жирность и эталонную структуру.

    Args:
        text: Текст блока.
        font_size: Размер шрифта блока.
        base_font: Базовый размер шрифта документа.
        is_bold: Флаг жирного шрифта.
        title_found: Флаг наличия главного заголовка статьи.
        in_references: Флаг нахождения в блоке списка литературы.

    Returns:
        Кортеж (is_heading, level, number).
    """
    text = text.strip()
    text_lower = text.lower()

    if in_references:
        if not re.match(r"^(appendix|supplementary|section\s+s|s\d+)", text_lower):
            return False, 0, None

    if re.match(r"^(figure|fig\.|table|scheme)\s*\d+", text_lower):
        return False, 0, None

    if text_lower in STANDARD_SECTION_ONTOLOGY:
        return True, 2, None

    if len(text) < 4 or len(text) > 300:
        return False, 0, None

    if re.fullmatch(
        r"[\d\s\.\,\-]+(?:mM|cm|mm|mV|mA|h|min|s|kΩ|µm)?", text, re.IGNORECASE
    ):
        return False, 0, None

    # Поиск нумерации
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

            # В заголовках редко бывает много запятых
            # Запятые в начале строки часто указывают на список авторов или аффилиации
            if text.count(",") >= 2:
                return False, 0, None

            if not re.match(r"^[A-Z]\.\s+[A-Z]", rest_text):
                level = len(number_str.split(".")) + 1
                return True, level, number_str

    # Эвристики по шрифту
    if font_size > base_font + 2.5:
        if not title_found:
            return True, 1, None
    elif is_bold and font_size >= base_font - 0.5:
        if not text.endswith(".") and len(text) < 100 and "\n" not in text:
            # Вырезаем спецсимволы в начале, чтобы проверить первую букву
            clean_start = re.sub(r"^[^a-zA-Z]+", "", text)

            # Заголовки разделов в большинстве случаев начинаются с заглавной буквы
            is_lower_start = bool(clean_start and clean_start[0].islower())

            has_too_much_punct = (text.count(",") >= 2) or (";" in text)

            if not is_lower_start and not has_too_much_punct:
                if len(text.split()) > 1 or text_lower in STANDARD_SECTION_ONTOLOGY:
                    if not text_lower.startswith("abstract:"):
                        return True, 2, None

    return False, 0, None


def build_section_tree(blocks: List[PageBlock]) -> List[Section]:
    """
    Собирает плоский список текстовых блоков в рекурсивное дерево секций.

    Args:
        blocks: Отсортированный по порядку чтения список блоков от экстрактора.

    Returns:
        Список корневых объектов Section (уровень 1) с вложенными подсекциями.
    """
    if not blocks:
        return []

    base_font = _get_base_font_size(blocks)
    root_sections: List[Section] = []
    stack: List[Section] = []

    current_section = None
    title_found = False
    in_references = False

    for block in blocks:
        if block.block_type == "image":
            continue

        text = block.text.strip() if block.text else ""
        if not text:
            continue

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

                new_section = Section(
                    heading=chunk_text,
                    level=level,
                    content="",
                    subsections=[],
                    number=number,
                    status=None,
                    status_effective_from=None,
                )

                while stack and stack[-1].level >= level:
                    stack.pop()

                if not stack:
                    root_sections.append(new_section)
                else:
                    stack[-1].subsections.append(new_section)

                stack.append(new_section)
                current_section = new_section
            else:
                # Если мы еще не встретили ни одного заголовка, весь текст
                # до первого H1/H2 считаем метаданными или абстрактом
                if not current_section:
                    current_section = Section(
                        heading="Metadata/Abstract",
                        level=1,
                        content="",
                        subsections=[],
                        number=None,
                        status=None,
                        status_effective_from=None,
                    )
                    root_sections.append(current_section)
                    stack.append(current_section)

                if current_section.content:
                    current_section.content += f"\n\n{chunk_text}"
                else:
                    current_section.content = chunk_text

    return root_sections


def extract_acknowledgments(sections: List[Section]) -> Optional[str]:
    """
    Извлекает текст раздела благодарностей из собранного дерева секций.

    Ищет секции с заголовками, соответствующими вариациям "Acknowledgements"
    или "Acknowledgments" или "Funding". Если находит, вырезает текст,
    удаляет саму секцию из дерева (для избежания дублирования данных)
    и возвращает строку.

    Args:
        sections: Список корневых секций (дерево).

    Returns:
        Текст раздела благодарностей единой строкой или None, если он отсутствует.
    """
    target_keywords = {"acknowledgements", "acknowledgments", "funding"}

    for i, section in enumerate(sections):
        # Проверяем корневые секции
        if section.heading.strip().lower() in target_keywords:
            ack_text = section.content.strip()
            sections.pop(i)
            return ack_text if ack_text else None

        # Проверяем подсекции
        for j, sub in enumerate(section.subsections):
            if sub.heading.strip().lower() in target_keywords:
                ack_text = sub.content.strip()
                section.subsections.pop(j)
                return ack_text if ack_text else None

    return None
