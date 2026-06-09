#!/bin/bash
set -e

echo "1. Статический анализ (Ruff)"
uv run ruff check .

echo "2. Проверка типов (Mypy)"
uv run mypy

echo "3. Запуск готовых тестов"
uv run pytest tests/ -v --cov=src --cov-report=term-missing

echo "Проверки пройдены!"
exec "$@"