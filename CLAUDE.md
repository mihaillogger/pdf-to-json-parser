# CLAUDE.md

Проект: парсер PDF научных статей (химия) → строгий машиночитаемый JSON по схеме ТЗ.
CLI, батч-обработка ~100 документов. Python ≥ 3.10, кросс-платформенность (Linux/macOS/Windows).

**Полные правила процесса — в [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Прочитай их перед изменениями.**

## Железные правила (нарушение ломает пайплайн у всей команды)

- **Зависимости — только `uv`.** Никогда `pip install`. Новый пакет → `uv add <пакет>` (сам обновит `pyproject.toml` и `uv.lock`). `uv.lock` всегда коммитится. `pyproject.toml`/`uv.lock` руками не править.
- **Ветки.** Никогда не пушить напрямую в `main`. Только `feat/<имя>` или `fix/<имя>`, отпочкованные от свежего `main` (`git checkout main && git pull && git checkout -b feat/...`). Имена вида `boban-test` запрещены.
- **Не коммить и не пушь без явной просьбы пользователя.**
- **Pydantic-схемы.** Модули не возвращают сырые `dict` — только модели из [src/parser/schemas.py](src/parser/schemas.py) (`Section`, `Figure`, `Table`, `Equation`, `Metadata`, `Document`). Очистка текста (strip, висячие переносы) зашита в базовый класс — не дублировать. Имена/типы полей JSON менять нельзя.
- **Проверки перед PR:** `uv run ruff check .` и `uv run mypy src/` должны быть зелёными. Type hints + docstrings для публичного API.
- **Внешние API (CrossRef/OpenAlex) и LLM — только опционально, отключаемые флагом CLI.** Обязателен оффлайн-режим. Запрещён hard-coding под конкретные журналы и ручная правка тестовых PDF.

## Окружение

- `uv` установлен (Homebrew). `.venv` поднят через `uv sync`.
- Запуск кода: `uv run python -m parser ...` или `uv run <команда>`.
- ⚠️ Текущий `.venv` на Python 3.14 — очень свежий; часть библиотек (PyMuPDF, camelot, OCR/ML) может не иметь wheel'ов. Если `uv add` падает на сборке — вероятная причина в версии Python.

## Архитектура (слои, тестируемые независимо)

`cli.py` (typer) · `schemas.py` (стандарт JSON) · извлечение текста (PyMuPDF) · `metadata.py` · `sections.py` · `figures.py` · `equations.py`. Оркестратор собирает объекты схем в финальный `Document`.
