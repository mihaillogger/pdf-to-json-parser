#!/bin/bash
set -e

echo "1. Статический анализ (Ruff)"
uv run ruff check .

echo "2. Проверка типов (Mypy)"
uv run mypy

echo "3. Запуск готовых тестов"
uv run pytest tests/test_metadata.py tests/test_sections.py -v --cov=src --cov-report=term-missing
uv run pytest tests/test_integration.py -v --cov=src --cov-report=term-missing
# после того как все напишут тесты заменить на:
# uv run pytest tests/ -v --cov=src --cov-report=term-missing

echo "Проверки пройдены!"
exec "$@"