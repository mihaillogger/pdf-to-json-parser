from unittest.mock import MagicMock, patch

import pytest

from parser.extractor import PDFExtractor


@pytest.fixture
def mock_page() -> MagicMock:
    """Фикстура для создания базового мока страницы fitz."""
    page = MagicMock()
    page.rect.width = 600.0
    page.rect.height = 800.0
    return page


@pytest.fixture
def extractor() -> PDFExtractor:
    """Фикстура для инициализации экстрактора с дефолтными настройками."""
    return PDFExtractor(pdf_path="dummy.pdf")


def test_extract_empty_pdf(extractor: PDFExtractor) -> None:
    """Проверяет корректную работу, если в PDF нет блоков."""
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert result == []


def test_extract_filters_y_axis_noise(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет отсечение верхних (y0 < 50) и нижних (y1 > 750) колонтитулов."""
    raw_blocks = [
        {
            "type": 0,
            "bbox": [50, 20, 200, 45],  # Верхний колонтитул (y0 = 20 < 50)
            "lines": [
                {
                    "bbox": [50, 20, 200, 45],
                    "spans": [
                        {"text": "Header Text", "size": 10.0, "bbox": [50, 20, 200, 45]}
                    ],
                }
            ],
        },
        {
            "type": 0,
            "bbox": [50, 100, 200, 150],  # Валидный контент
            "lines": [
                {
                    "bbox": [50, 100, 200, 150],
                    "spans": [
                        {
                            "text": "Valid Block Content",
                            "size": 10.0,
                            "bbox": [50, 100, 200, 150],
                        }
                    ],
                }
            ],
        },
        {
            "type": 0,
            "bbox": [50, 760, 200, 790],  # Нижний колонтитул (y1 = 790 > 750)
            "lines": [
                {
                    "bbox": [50, 760, 200, 790],
                    "spans": [
                        {"text": "Page 1", "size": 10.0, "bbox": [50, 760, 200, 790]}
                    ],
                }
            ],
        },
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()

        assert len(result) == 1
        assert result[0].text == "Valid Block Content"


def test_extract_filters_short_text_noise(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет эвристику удаления мусорных блоков длиной <= 3 символов."""
    raw_blocks = [
        {
            "type": 0,
            "bbox": [50, 100, 200, 150],
            "lines": [
                {
                    "bbox": [50, 100, 200, 150],
                    # Мусор
                    "spans": [
                        {"text": "10", "size": 10.0, "bbox": [50, 100, 200, 150]}
                    ],
                }
            ],
        },
        {
            "type": 0,
            "bbox": [50, 200, 200, 250],
            "lines": [
                {
                    "bbox": [50, 200, 200, 250],
                    "spans": [
                        {
                            "text": "Normal text entry",
                            "size": 10.0,
                            "bbox": [50, 200, 200, 250],
                        }
                    ],
                }
            ],
        },
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert len(result) == 1
        assert result[0].text == "Normal text entry"


def test_extract_image_block(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет корректное распознавание графических блоков (type == 1)."""
    raw_blocks = [
        {
            "type": 1,
            "bbox": [100, 150, 300, 400],
        }
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert len(result) == 1
        assert result[0].block_type == "image"
        assert result[0].text is None
        assert result[0].bbox.left == 100
        assert result[0].bbox.bottom == 400


def test_span_horizontal_spacing_injection(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет вставку пробела, если расстояние между спанами по X > 4.0."""
    raw_blocks = [
        {
            "type": 0,
            "bbox": [50, 100, 300, 150],
            "lines": [
                {
                    "bbox": [50, 100, 300, 150],
                    "spans": [
                        {
                            "text": "LeftColumn",
                            "size": 10.0,
                            "bbox": [50, 100, 100, 120],
                        },
                        # Смещение 5.0 > 4.0
                        {
                            "text": "RightColumn",
                            "size": 10.0,
                            "bbox": [105, 100, 200, 120],
                        },
                    ],
                }
            ],
        }
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert result[0].text == "LeftColumn RightColumn"


def test_block_splitting_on_font_size_change(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет разделение единого блока на два при изменении размера шрифта > 1.0."""
    raw_blocks = [
        {
            "type": 0,
            "bbox": [50, 100, 300, 250],
            "lines": [
                {
                    "bbox": [50, 100, 200, 120],
                    "spans": [
                        {
                            "text": "Large Header Title",
                            "size": 14.0,
                            "bbox": [50, 100, 200, 120],
                        }
                    ],
                },
                {
                    "bbox": [50, 130, 200, 150],
                    "spans": [
                        {
                            "text": "Regular paragraph text",
                            "size": 10.0,
                            "bbox": [50, 130, 200, 150],
                        }
                    ],
                },
            ],
        }
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert len(result) == 2
        assert result[0].text == "Large Header Title"
        assert result[0].font_size == 14.0
        assert result[1].text == "Regular paragraph text"
        assert result[1].font_size == 10.0


def test_block_splitting_on_bold_flag_change(
    extractor: PDFExtractor, mock_page: MagicMock
) -> None:
    """Проверяет разделение блока при одинаковом шрифте, но смене жирности."""
    raw_blocks = [
        {
            "type": 0,
            "bbox": [50, 100, 300, 250],
            "lines": [
                {
                    "bbox": [50, 100, 200, 120],
                    "spans": [
                        {
                            "text": "Bold Section Header",
                            "size": 10.0,
                            "flags": 16, # Бит жирности взведен
                            "font": "Helvetica-Bold",
                            "bbox": [50, 100, 200, 120]
                        }
                    ],
                },
                {
                    "bbox": [50, 130, 200, 150],
                    "spans": [
                        {
                            "text": "Standard description text",
                            "size": 10.0,
                            "flags": 0,
                            "font": "Helvetica",
                            "bbox": [50, 130, 200, 150]
                        }
                    ],
                },
            ],
        }
    ]
    mock_page.get_text.return_value = {"blocks": raw_blocks}

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        result = extractor.extract()
        assert len(result) == 2
        assert result[0].text == "Bold Section Header"
        assert result[0].is_bold is True
        assert result[1].text == "Standard description text"
        assert result[1].is_bold is False
