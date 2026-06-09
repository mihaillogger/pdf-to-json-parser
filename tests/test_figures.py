"""Юнит-тесты для модуля извлечения визуальных элементов."""

from unittest.mock import MagicMock, patch

from parser.figures import SpatialExtractor, VLMTableExtractor


class TestVLMTableExtractor:
    """Набор тестов для компонента VLMTableExtractor."""

    def test_extract_2d_array_file_not_found(self) -> None:
        """Граничный тест: файл изображения не существует.

        Ожидаемое поведение: немедленный возврат пустого списка.
        """
        extractor = VLMTableExtractor()
        assert extractor.extract_2d_array("non_existent.png") == []

    @patch("parser.figures.os.path.exists", return_value=True)
    @patch("parser.figures.ollama.chat")
    def test_extract_2d_array_success(
        self, mock_chat: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Юнит-тест: валидный ответ от мультимодальной нейросети.

        Ожидаемое поведение: успешный парсинг 2D-массива.
        """
        mock_chat.return_value = {
            "message": {
                "content": '{"data": [["1", "2"], ["3", "4"]]}'
            }
        }
        extractor = VLMTableExtractor()
        result = extractor.extract_2d_array("dummy.png")
        assert result == [["1", "2"], ["3", "4"]]

    @patch("parser.figures.os.path.exists", return_value=True)
    @patch("parser.figures.ollama.chat")
    def test_extract_2d_array_1d_fallback(
        self, mock_chat: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Граничный тест: защита от деструктуризации массива.

        Ожидаемое поведение: плоский 1D-список оборачивается в 2D.
        """
        mock_chat.return_value = {
            "message": {
                "content": '{"data": ["A", "B", "C"]}'
            }
        }
        extractor = VLMTableExtractor()
        result = extractor.extract_2d_array("dummy.png")
        assert result == [["A"], ["B"], ["C"]]

    @patch("parser.figures.os.path.exists", return_value=True)
    @patch("parser.figures.ollama.chat")
    def test_extract_2d_array_type_cast(
        self, mock_chat: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Граничный тест: защита от нарушения типов (float вместо str).

        Ожидаемое поведение: принудительный кастинг чисел в строки.
        """
        mock_chat.return_value = {
            "message": {
                "content": '{"data": [[1.5, 2], [3.0, 4]]}'
            }
        }
        extractor = VLMTableExtractor()
        result = extractor.extract_2d_array("dummy.png")
        assert result == [["1.5", "2"], ["3.0", "4"]]

    @patch("parser.figures.os.path.exists", return_value=True)
    @patch("parser.figures.ollama.chat")
    def test_extract_2d_array_server_error(
        self, mock_chat: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Граничный тест: критический сбой локального сервера Ollama.

        Ожидаемое поведение: перехват исключения и возврат пустого списка.
        """
        mock_chat.side_effect = Exception("502 Bad Gateway")
        extractor = VLMTableExtractor()
        result = extractor.extract_2d_array("dummy.png")
        assert result == []


class TestSpatialExtractor:
    """Набор тестов для оркестратора визуальных элементов SpatialExtractor."""

    @patch("parser.figures.hf_hub_download")
    @patch("parser.figures.YOLOv10")
    @patch("parser.figures.os.makedirs")
    def test_initialization(
        self, mock_makedirs: MagicMock, mock_yolo: MagicMock, mock_hub: MagicMock
    ) -> None:
        """Проверка инициализации оркестратора и создания директорий."""
        extractor = SpatialExtractor("test_dir")
        mock_makedirs.assert_called_with("test_dir", exist_ok=True)
        assert extractor.vlm.model_name == "llava"

    @patch("parser.figures.hf_hub_download")
    @patch("parser.figures.YOLOv10")
    @patch("parser.figures.fitz.open")
    def test_extract_visuals_empty(
        self, mock_fitz: MagicMock, mock_yolo: MagicMock, mock_hub: MagicMock
    ) -> None:
        """Граничный тест: обработка страницы без целевых объектов.

        Ожидаемое поведение: возврат пустых списков фигур и таблиц.
        """
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.return_value = mock_doc

        extractor = SpatialExtractor("test_dir")

        mock_yolo_res = MagicMock()
        mock_yolo_res.boxes = []
        extractor.model.predict.return_value = [mock_yolo_res]

        figures, tables = extractor.extract_visuals("dummy.pdf")

        assert len(figures) == 0
        assert len(tables) == 0
