"""
Интеграционные тесты для проверки конвейера сборки (parser.pipeline).
Проверяют совместную работу текстового экстрактора, метаданных, секций и схем.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from parser.pipeline import process_single_file
from parser.schemas import BBox, PageBlock


def test_full_pipeline_integration(tmp_path: Path) -> None:
    """Проверяет полный цикл обработки документа оркестратором.

    Создает фейковые блоки текста, прогоняет их через метаданные и сборку секций,
    подменяет вызовы тяжелых нейронок и проверяет валидность итогового JSON.
    """
    fake_pdf = tmp_path / "test_article.pdf"
    fake_pdf.touch()

    fake_blocks = [
        PageBlock(
            text="A Great Scientific Paper",
            font_size=18.0,
            bbox=BBox(left=0, top=10, right=100, bottom=20),
            page_number=1,
            block_type="text",
            is_bold=True,
        ),
        PageBlock(
            text="DOI: 10.1234/test.123",
            font_size=10.0,
            bbox=BBox(left=0, top=30, right=100, bottom=40),
            page_number=1,
            block_type="text",
            is_bold=False,
        ),
        PageBlock(
            text="1. Introduction",
            font_size=14.0,
            bbox=BBox(left=0, top=50, right=100, bottom=60),
            page_number=1,
            block_type="text",
            is_bold=True,
        ),
        PageBlock(
            text="Some intro text here.",
            font_size=12.0,
            bbox=BBox(left=0, top=70, right=100, bottom=80),
            page_number=1,
            block_type="text",
            is_bold=False,
        ),
    ]

    with (
        patch("parser.pipeline.get_page_blocks", return_value=fake_blocks),
        patch("parser.pipeline.extract_metadata") as mock_meta,
        patch("parser.pipeline.SpatialExtractor") as mock_spatial,
        patch("parser.pipeline.EquationExtractor") as _mock_eq,
    ):
        from parser.schemas import Metadata

        mock_meta.return_value = Metadata(
            title="A Great Scientific Paper",
            title_en=None,
            authors=["Doe, John"],
            abstract="Fake abstract",
            keywords=[],
            doi="10.1234/test.123",
            journal=None,
            year=2026,
            metadata_source="pdf",
            metadata_confidence=0.9,
            normative=None,
        )

        mock_spatial_instance = MagicMock()
        mock_spatial_instance.extract_visuals.return_value = ([], [])
        mock_spatial.return_value = mock_spatial_instance

        process_single_file(
            pdf_path=fake_pdf,
            output_dir=tmp_path,
            overwrite=True,
            offline=True,  # Без сети
            use_crossref=False,  # Без API
            use_llm=False,  # Без локальной Ollama
            extract_images=False,  # Без картинок
        )

    json_path = tmp_path / "test_article.json"

    assert json_path.exists(), "JSON файл не был создан конвейером"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["title"] == "A Great Scientific Paper"
    assert data["metadata"]["doi"] == "10.1234/test.123"

    assert len(data["sections"]) == 1
    assert data["sections"][0]["heading"] == "1. Introduction"
    assert "Some intro text" in data["sections"][0]["content"]
    assert data["raw_text"] is not None
