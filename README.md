# PDF → JSON: парсер научных статей по химии

CLI-инструмент для пакетного извлечения структурированных данных из научных
PDF-статей (химия) в строгий машиночитаемый **JSON по схеме ТЗ**. Рассчитан на
батч-обработку ~100 документов, кросс-платформенный (Linux / macOS / Windows),
Python ≥ 3.10.

Парсер вытаскивает из каждого PDF:
- **метаданные** — заголовок, авторы, DOI, год, журнал, аннотация, ключевые слова;
- **секции** — иерархическое дерево разделов с заголовками и текстом;
- **фигуры** — координаты, кропы изображений, подписи (caption) и id;
- **таблицы** — кропы, подписи и распознанные данные (2D-массив);
- **уравнения** — LaTeX, номер, координаты и текстовый контекст;
- **полный текст** документа.

---

## Возможности

- **Три режима работы** под разные условия (см. [Режимы](#режимы-работы)):
  онлайн (CrossRef), офлайн + локальная LLM, офлайн без LLM.
- **Оффлайн по умолчанию** — внешние API (CrossRef/OpenAlex) и LLM строго
  опциональны и отключаются флагами.
- **OCR-фоллбэк** — для сканов без текстового слоя (Tesseract, `eng+rus`).
- **Параллельная батч-обработка** директории (`--workers`).
- **Логирование** статуса по каждому документу + итоговая сводка (в stdout и `run.log`).
- **Строгие Pydantic-схемы** — гарантированно валидный JSON.
- **Метрики качества** (precision / recall / F1) на отложенной выборке.

---

## Архитектура

Модульная, слои тестируются независимо. Оркестратор собирает результат в объект
`Document` и сериализует в JSON.

| Модуль | Назначение |
|---|---|
| [`cli.py`](src/parser/cli.py) | CLI на Typer: парсинг аргументов, режимы, логирование |
| [`pipeline.py`](src/parser/pipeline.py) | Оркестратор: полный цикл обработки + статусы/сводка |
| [`schemas.py`](src/parser/schemas.py) | Pydantic-схемы итогового JSON (стандарт данных) |
| [`extractor.py`](src/parser/extractor.py) | Извлечение текстовых блоков с учётом колонок (PyMuPDF) |
| [`ocr.py`](src/parser/ocr.py) | OCR-фоллбэк для сканов (Tesseract) |
| [`metadata.py`](src/parser/metadata.py) | Каскад метаданных: DOI → CrossRef → LLM → эвристики |
| [`sections.py`](src/parser/sections.py) | Построение дерева секций |
| [`figures.py`](src/parser/figures.py) | Фигуры/таблицы (DocLayout-YOLO + VLM для таблиц) |
| [`equations.py`](src/parser/equations.py) | Уравнения (YOLO + Pix2Tex → LaTeX) |
| [`evaluation.py`](src/parser/evaluation.py) | Метрики качества precision/recall/F1 |

---

## Требования

- **Python 3.10–3.12** (на 3.13+ часть ML-зависимостей не имеет колёс).
- **[uv](https://docs.astral.sh/uv/)** — менеджер зависимостей и окружения.
- Системные пакеты (для OCR и обработки изображений/таблиц):

```bash
# macOS
brew install tesseract tesseract-lang ghostscript poppler

# Debian/Ubuntu
sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
                    ghostscript poppler-utils libgl1 libglib2.0-0
```

---

## Установка

```bash
git clone https://github.com/mihaillogger/pdf-to-json-parser.git
cd pdf-to-json-parser
uv sync          # поднимет .venv и поставит зависимости строго по uv.lock
```

> Зависимости меняются только через `uv add <пакет>`; `uv.lock` коммитится.

---

## Использование

Точка входа — пакет `parser`:

```bash
# один файл
uv run python -m parser --input article.pdf --output out/

# целая директория, 4 параллельных воркера
uv run python -m parser --input ./pdfs --output out/ --workers 4
```

### Флаги CLI

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--input` | — | PDF-файл или директория с PDF |
| `--output` | — | Папка для JSON, изображений и `run.log` |
| `--workers` | `1` | Число параллельных процессов для батча |
| `--overwrite` | off | Перезаписывать уже существующие JSON |
| `--log-level` | `INFO` | Уровень логов (`INFO` / `DEBUG`) |
| `--offline` | off | Полностью отключить сеть (принудительный локальный режим) |
| `--crossref / --no-crossref` | on | Использовать CrossRef API для метаданных |
| `--llm / --no-llm` | on | Использовать локальную LLM для добора метаданных |
| `--extract-images` | on | Извлекать фигуры/таблицы (YOLO + VLM) |

### Режимы работы

| Режим | Команда | Что использует |
|---|---|---|
| **Онлайн** | по умолчанию (CrossRef включён) | CrossRef по DOI — максимальное качество метаданных |
| **Офлайн + LLM** | `--offline` | Локальная модель через Ollama, без сети |
| **Офлайн без LLM** | `--offline --no-llm` | Только эвристики, полностью автономно |

**Локальная LLM (Ollama)** для офлайн-режима:

```bash
# установить Ollama и подтянуть модели
ollama pull qwen2.5:3b   # добор метаданных
ollama pull llava        # распознавание таблиц (VLM)
```

Адрес и модель настраиваются переменными окружения (важно для Docker):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Эндпоинт LLM для метаданных |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Модель для метаданных |
| `OLLAMA_HOST` | `http://localhost:11434` | Хост Ollama для VLM-таблиц (`llava`) |

---

## Формат вывода

На каждый `<имя>.pdf` создаётся `<имя>.json` (объект `Document`), кропы фигур/таблиц
в `out/images/<имя>/`, а также общий `out/run.log`.

```jsonc
{
  "metadata": {
    "title": "...", "title_en": null, "authors": ["Фамилия, И."],
    "abstract": "...", "keywords": [], "doi": "10.1039/...",
    "journal": "...", "year": 2023,
    "metadata_source": "crossref", "metadata_confidence": 0.95
  },
  "sections":  [{ "heading": "Introduction", "level": 1, "content": "...",
                  "subsections": [], "number": "1" }],
  "figures":   [{ "id": "Figure 1", "caption": "...", "page": 2,
                  "bbox": {}, "img_path": "images/.../fig_1_p2.png" }],
  "tables":    [{ "id": "Table 1", "caption": "...", "data": [["..."]] }],
  "equations": [{ "id": "(1)", "latex": "E = mc^2", "context": "..." }],
  "acknowledgments": null,
  "raw_text": "полный текст документа"
}
```

---

## Метрики качества

Оценка precision / recall / F1 на отложенной выборке относительно эталона
`evaluation/gold.json` (CrossRef по DOI + ручная проверка). Подробности и
текущие результаты — в [`evaluation/README.md`](evaluation/README.md).

```bash
# метрики по метаданным (извлечение «на лету»)
uv run python scripts/eval_metadata.py --input ./pdfs --mode offline

# метрики по полному выходу пайплайна (Document-JSON)
uv run python -m parser --input ./pdfs --output out/
uv run python scripts/eval_pipeline.py --pred-dir out/ --gold evaluation/gold.json
```

---

## Веб-интерфейс (Streamlit)

Графическая обёртка для загрузки PDF и просмотра результата без командной строки.

```bash
uv run streamlit run <app>.py
```

> Раздел в активной разработке; точная команда запуска появится вместе с интерфейсом.

---

## Docker

Для воспроизводимого запуска без локальной установки зависимостей.

```bash
docker build -t pdf-parser .
docker run --rm \
  -v "$PWD/pdfs:/data/in" -v "$PWD/out:/data/out" \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  pdf-parser --input /data/in --output /data/out
```

Ключевые моменты сборки:
- базовый образ **`python:3.12-slim`** (на 3.13+ ломаются ML-колёса);
- зависимости — `uv sync --frozen` (не регенерировать lock);
- системные пакеты из раздела [Требования](#требования);
- Ollama снаружи контейнера → адресуется через `host.docker.internal`
  (переменные `OLLAMA_URL` / `OLLAMA_HOST`);
- для офлайна без LLM достаточно `--offline --no-llm` — внешние сервисы не нужны.

> Dockerfile и веб-интерфейс готовятся в отдельной ветке.

---

## Разработка

```bash
uv run ruff check .          # линтер
uv run mypy src/ tests/      # типы (strict)
uv run pytest                # тесты
```

Перед PR обе проверки (`ruff`, `mypy`) должны быть зелёными. Ветки — только
`feat/<имя>` или `fix/<имя>` от свежего `main`; напрямую в `main` не пушим.

---

## Структура проекта

```
src/parser/        модули парсера (см. Архитектуру)
scripts/           скрипты оценки качества (метрики)
evaluation/        эталонная разметка (gold.json) + отчёт по метрикам
tests/             модульные тесты
pyproject.toml     зависимости и конфиг инструментов
uv.lock            зафиксированные версии (коммитится)
```
