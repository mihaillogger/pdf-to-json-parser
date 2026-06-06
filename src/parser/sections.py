import re
from collections import Counter
from typing import List, Optional, Tuple

from parser.schemas import PageBlock, Section

# LEXICAL ONTOLOGY (Фоллбэк для сломанных PDF)
# Используется как запасной механизм определения секций, если
# PDF-документ потерял метаданные жирности шрифта (subset fonts issue).
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
    "device design and validation",
    "methods",
}

# Ключевые слова для отсечения мусорных колонтитулов и врезок
SKIP_HEADER_KEYWORDS = {
    "graphical abstract",
    "highlights",
    "supporting information",
    "full paper",
    "section s",
    "table of contents",
    "published online",
    "doi:",
    "corrigendum",
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
        if getattr(b, "block_type", "text") == "text" and b.font_size is not None
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
    Анализирует текстовый блок и определяет, является ли он заголовком,
    опираясь на размер шрифта, жирность и паттерны нумерации.

    Args:
        text: Текст блока.
        font_size: Размер шрифта блока.
        base_font: Базовый размер шрифта документа.
        is_bold: Флаг жирного шрифта (отдает экстрактор).

    Returns:
        Кортеж (is_heading, level, number).
    """
    text = text.strip()
    text_lower = text.lower()

    # 0. Лексический фоллбэк
    # Срабатывает, если заголовок стандартный, но PDF не отдал флаг жирности
    if text_lower in STANDARD_SECTION_ONTOLOGY:
        return True, 2, None

    # 1. Жесткий фильтр мусора: слишком короткие/длинные блоки или оси графиков
    if len(text) < 4 or len(text) > 300:
        return False, 0, None

    # Отсекаем строки, состоящие только из цифр и единиц измерения (оси)
    if re.fullmatch(
        r"[\d\s\.\,\-]+(?:mM|cm|mm|mV|mA|h|min|s|kΩ|µm)?", text, re.IGNORECASE
    ):
        return False, 0, None

    # 2. Ищем нумерацию (ОТКЛЮЧАЕТСЯ внутри списка литературы)
    if not in_references:
        match = re.match(r"^((?:\d+\.)+\d*|\d+\.?|[IVX]+\.)\s+(.+)", text)
        if match:
            number_str = match.group(1).strip(".")
            rest_text = match.group(2)

            if (
                len(text) < 100
                and "\n" not in text
                and len(rest_text) > 2
                and any(c.isalpha() for c in rest_text)
            ):
                if (
                    not re.match(r"^[A-Z]\.\s+[A-Z]", rest_text)
                    and "département" not in rest_text.lower()
                ):
                    level = len(number_str.split(".")) + 1
                    return True, level, number_str

    # 3. Эвристики по шрифту и жирности
    if font_size > base_font + 2.5:
        # Разрешаем уровень 1, только если название статьи еще не найдено
        if not title_found:
            return True, 1, None
    elif is_bold and font_size >= base_font - 0.5:
        if not text.endswith(".") and len(text) < 100 and "\n" not in text:
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
        if getattr(block, "block_type", "text") == "image":
            continue

        text = block.text.strip() if block.text else ""
        if not text:
            continue

        lines = text.split("\n", 1)
        if len(lines) == 2 and lines[0].strip().lower() in STANDARD_SECTION_ONTOLOGY:
            chunks = [
                (lines[0].strip(), block.font_size, getattr(block, "is_bold", False)),
                (lines[1].strip(), block.font_size, False),
            ]
        else:
            chunks = [(text, block.font_size, getattr(block, "is_bold", False))]

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
