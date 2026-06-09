"""Юнит-тесты для модуля извлечения уравнений."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parser.equations import EquationExtractor


@pytest.fixture
def mock_yolo_weights(tmp_path: Path) -> str:
    """Создает фейковый файл весов для успешной инициализации класса."""
    weights_path = tmp_path / "best.pt"
    weights_path.touch()
    return str(weights_path)


class TestEquationExtractor:
    """Тестовый набор для класса EquationExtractor."""

    @patch("parser.equations.YOLO")
    @patch("parser.equations.LatexOCR")
    def test_initialization_success(
        self, mock_ocr: MagicMock, mock_yolo: MagicMock, mock_yolo_weights: str
    ) -> None:
        """Проверка успешной загрузки моделей."""
        extractor = EquationExtractor(mock_yolo_weights)
        assert extractor.model is not None
        assert extractor.math_ocr is not None

    def test_initialization_file_not_found(self) -> None:
        """Проверка граничного условия: отсутствие весов вызывает ошибку."""
        with pytest.raises(FileNotFoundError):
            EquationExtractor("fake_path.pt")

    @patch("parser.equations.YOLO")
    def test_clean_latex(self, mock_yolo: MagicMock, mock_yolo_weights: str) -> None:
        """Проверка алгоритма очистки визуального мусора в LaTeX."""
        extractor = EquationExtractor(mock_yolo_weights)
        raw = "E=mc^2 \\quad \\tag{1}"
        cleaned = extractor._clean_latex(raw)
        assert cleaned == "E=mc^2 \\tag{1}"

    @patch("parser.equations.YOLO")
    def test_extract_id(self, mock_yolo: MagicMock, mock_yolo_weights: str) -> None:
        """Проверка извлечения идентификаторов из разных форматов."""
        extractor = EquationExtractor(mock_yolo_weights)
        assert extractor._extract_id("a^2 + b^2 = c^2 (5)") == "(5)"
        assert extractor._extract_id("x = y \\tag{12}") == "(12)"
        assert extractor._extract_id("x = y") is None

    @patch("parser.equations.YOLO")
    @patch("parser.equations.fitz.open")
    @patch("parser.equations.Image.open")
    def test_process_pdf_empty_boxes(
        self,
        mock_image_open: MagicMock,
        mock_fitz: MagicMock,
        mock_yolo: MagicMock,
        mock_yolo_weights: str,
    ) -> None:
        """Граничный тест: YOLO не нашла ни одной формулы."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_fitz.return_value.__enter__.return_value = mock_doc

        extractor = EquationExtractor(mock_yolo_weights)

        mock_yolo_res = MagicMock()
        mock_yolo_res.boxes = []
        extractor.model.predict.return_value = [mock_yolo_res]

        results = extractor.process_pdf("dummy.pdf")
        assert len(results) == 0
