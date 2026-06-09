"""Тесты модуля сборки иерархии секций (parser.sections)"""

from parser.schemas import BBox, PageBlock, Section
from parser.sections import (
    _analyze_heading,
    _get_base_font_size,
    build_section_tree,
    extract_acknowledgments,
)


def _block(
    text: str | None,
    font_size: float | None = 12.0,
    is_bold: bool = False,
    block_type: str = "text",
) -> PageBlock:
    """Создает мок-объект PageBlock для тестов"""
    return PageBlock(
        text=text,
        font_size=font_size,
        bbox=BBox(left=50.0, top=100.0, right=500.0, bottom=120.0),
        page_number=1,
        block_type=block_type,
        is_bold=is_bold,
    )


# Тесты для _get_base_font_size


def test_get_base_font_size_empty() -> None:
    assert _get_base_font_size([]) == 12.0


def test_get_base_font_size_most_common() -> None:
    blocks = [
        _block("Title", font_size=18.0),
        _block("text 1", font_size=10.5),
        _block("text 2", font_size=10.5),
        _block("text 3", font_size=10.5),
        _block("Heading", font_size=14.0),
    ]
    assert _get_base_font_size(blocks) == 10.5


def test_get_base_font_size_ignores_images() -> None:
    blocks = [
        _block("text 1", font_size=11.0),
        _block("", font_size=None, block_type="image"),
        _block("", font_size=None, block_type="image"),
    ]
    assert _get_base_font_size(blocks) == 11.0


# Тесты для _analyze_heading


def test_analyze_heading_standard_ontology() -> None:
    is_heading, level, num = _analyze_heading(
        text="Introduction",
        font_size=12.0,
        base_font=12.0,
        is_bold=True,
        title_found=True,
        in_references=False,
    )
    assert is_heading is True
    assert level == 2
    assert num is None


def test_analyze_heading_large_font_h1() -> None:
    is_heading, level, num = _analyze_heading(
        text="A Huge Title",
        font_size=16.0,
        base_font=10.0,
        is_bold=True,
        title_found=False,
        in_references=False,
    )
    assert is_heading is True
    assert level == 1
    assert num is None


def test_analyze_heading_numbered_section() -> None:
    is_heading, level, num = _analyze_heading(
        text="2.1. Experimental Setup",
        font_size=12.0,
        base_font=12.0,
        is_bold=True,
        title_found=True,
        in_references=False,
    )
    assert is_heading is True
    assert level == 3  # "2.1" бьется на 2 части + 1 = 3
    assert num == "2.1"


def test_analyze_heading_ignores_figures_and_tables() -> None:
    is_heading, level, num = _analyze_heading(
        text="Figure 1. The diagram of the process.",
        font_size=10.0,
        base_font=10.0,
        is_bold=True,
        title_found=True,
        in_references=False,
    )
    assert is_heading is False


def test_analyze_heading_handles_references_lock() -> None:
    is_heading, level, num = _analyze_heading(
        text="Some Random Bold Text",
        font_size=12.0,
        base_font=12.0,
        is_bold=True,
        title_found=True,
        in_references=True,  # Парсер внутри списка литературы
    )
    assert is_heading is False


# Тесты для build_section_tree


def test_build_section_tree_empty() -> None:
    assert build_section_tree([]) == []


def test_build_section_tree_skips_metadata_garbage() -> None:
    blocks = [
        _block("John Doe, ITMO University", font_size=10.0),  # Мусор
        _block("Received: 10 May", font_size=10.0),  # Мусор
        _block("1. Introduction", font_size=14.0, is_bold=True),
        _block("Actual start of the paper.", font_size=12.0),
    ]

    tree = build_section_tree(blocks)

    assert len(tree) == 1
    assert tree[0].heading == "1. Introduction"
    assert tree[0].content == "Actual start of the paper."
    assert "John Doe" not in tree[0].content


def test_build_section_tree_hierarchy() -> None:
    blocks = [
        _block("1. Introduction", font_size=14.0, is_bold=True),  # Станет level 2
        _block("Intro content.", font_size=10.0),
        _block("1.1. Background", font_size=12.0, is_bold=True),  # Станет level 3
        _block("Background content.", font_size=10.0),
        _block("2. Methods", font_size=14.0, is_bold=True),  # Станет level 2
        _block("Methods content.", font_size=10.0),
    ]

    tree = build_section_tree(blocks)

    # У нас должно быть 2 корневые секции (на 2-м уровне)
    assert len(tree) == 2

    # Первая секция (1. Introduction)
    assert tree[0].heading == "1. Introduction"
    assert tree[0].level == 2
    assert tree[0].content == "Intro content."
    assert len(tree[0].subsections) == 1

    # Подсекция внутри первой (1.1. Background)
    assert tree[0].subsections[0].heading == "1.1. Background"
    assert tree[0].subsections[0].level == 3
    assert tree[0].subsections[0].content == "Background content."

    # Вторая секция (2. Methods)
    assert tree[1].heading == "2. Methods"
    assert tree[1].level == 2
    assert tree[1].content == "Methods content."


# Тесты для extract_acknowledgments


def test_extract_acknowledgments_root_level() -> None:
    sections = [
        Section(
            heading="Introduction",
            level=1,
            content="Intro text",
            number=None,
            status=None,
            status_effective_from=None,
        ),
        Section(
            heading="Acknowledgements",
            level=1,
            content="Thanks to mom and dad.",
            number=None,
            status=None,
            status_effective_from=None,
        ),
        Section(
            heading="Conclusion",
            level=1,
            content="End of paper.",
            number=None,
            status=None,
            status_effective_from=None,
        ),
    ]

    ack_text = extract_acknowledgments(sections)

    assert ack_text == "Thanks to mom and dad."
    # Проверяем, что секция реально вырезалась из дерева
    assert len(sections) == 2
    assert sections[1].heading == "Conclusion"


def test_extract_acknowledgments_nested() -> None:
    sections = [
        Section(
            heading="Conclusion",
            level=1,
            content="End.",
            number=None,
            status=None,
            status_effective_from=None,
            subsections=[
                Section(
                    heading="Funding",
                    level=2,
                    content="Gazprom Neft paid for this.",
                    number=None,
                    status=None,
                    status_effective_from=None,
                )
            ],
        ),
    ]

    ack_text = extract_acknowledgments(sections)

    assert ack_text == "Gazprom Neft paid for this."
    # Проверяем, что подсекция удалилась, а родитель остался
    assert len(sections) == 1
    assert len(sections[0].subsections) == 0


def test_extract_acknowledgments_not_found() -> None:
    sections = [
        Section(
            heading="Introduction",
            level=1,
            content="Intro text",
            number=None,
            status=None,
            status_effective_from=None,
        )
    ]

    assert extract_acknowledgments(sections) is None
    assert len(sections) == 1
