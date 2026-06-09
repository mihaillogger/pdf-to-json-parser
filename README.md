# 📄 PDF → JSON: Парсер научных статей по химии

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

Проект представляет собой CLI-инструмент и Web-интерфейс для пакетного извлечения структурированных данных из научных PDF-статей.
Результат сохраняется в унифицированный JSON-формат по собственной схеме данных.

Основные объекты извлечения:
* **Метаданные** — заголовок, авторы, DOI, год, журнал, аннотация, ключевые слова.
* **Секции** — иерархическое дерево разделов и подзаголовков.
* **Фигуры** — координаты, изображения, подписи и ID.
* **Таблицы** — кропы, подписи и распознанные данные в виде 2D-таблицы.
* **Уравнения** — LaTeX, номер, координаты и контекст.
* **Полный текст** документа.

---

## ✨ Что умеет

* Три режима работы: онлайн через CrossRef, оффлайн с локальной LLM и оффлайн без нейросетей.
* Оффлайн-фоллбэк — сеть используется только при явном включении.
* OCR-поддержка для PDF без текстового слоя (Tesseract, `eng+rus`).
* Параллельная обработка через `--workers`.
* Детальное логирование и итоговая сводка по каждому запуску.
* Модельные Pydantic-схемы для валидации итогового JSON.
* Инструменты для оценки качества: Precision, Recall, F1.

---

## 🏗 Структура проекта

| Файл | Назначение |
| --- | --- |
| `src/parser/cli.py` | CLI: аргументы, режимы, логирование |
| `src/parser/pipeline.py` | Оркестратор обработки документа |
| `src/parser/schemas.py` | Pydantic-схемы выходных данных |
| `src/parser/extractor.py` | Извлечение текста с учётом колонок |
| `src/parser/ocr.py` | OCR-поддержка для сканов |
| `src/parser/metadata.py` | Каскад метаданных: DOI → CrossRef → LLM → эвристики |
| `src/parser/sections.py` | Построение дерева секций |
| `src/parser/figures.py` | Детекция фигур и таблиц |
| `src/parser/equations.py` | Распознавание уравнений |
| `src/parser/evaluation.py` | Метрики и оценка качества |

---

## 🚀 Быстрый старт с Docker и Streamlit

Лучший способ быстро запустить проект без локальной установки библиотек.

### 1. Установите Ollama
Проект использует локальные модели Ollama для дополнительных метаданных и распознавания таблиц.

```bash
ollama run qwen2.5:3b  # метаданные
ollama run llava       # таблицы
```

### 2. Запустите контейнер

```bash
git clone https://github.com/mihaillogger/pdf-to-json-parser.git
cd pdf-to-json-parser

docker-compose up --build -d
```

Откройте Web-интерфейс в браузере: **http://localhost:8501**.
Контейнер автоматически пробрасывает доступ к локальной Ollama через `host.docker.internal`.

---

## 💻 Локальная установка и запуск

### Требования

* Python 3.10–3.12
* [uv](https://docs.astral.sh/uv/)
* Системные зависимости: Tesseract, Poppler, Ghostscript, libgl1.

#### Debian/Ubuntu / WSL

```bash
sudo apt update && sudo apt install -y \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
  ghostscript poppler-utils libgl1 libglib2.0-0
```

#### macOS

```bash
brew install tesseract tesseract-lang ghostscript poppler
```

### Установка зависимостей

```bash
uv sync
```

### Запуск CLI

```bash
uv run python -m parser --input article.pdf --output out/
uv run python -m parser --input ./pdfs --output out/ --workers 4
```

### Основные флаги

| Флаг | Значение по умолчанию | Описание |
| --- | --- | --- |
| `--workers` | `1` | Число параллельных процессов |
| `--offline` | `off` | Отключает сеть |
| `--no-crossref` | `off` | Отключает CrossRef |
| `--no-llm` | `off` | Отключает локальную LLM |
| `--extract-images` | `on` | Включает извлечение фигур и таблиц |

---

## 📊 Что на выходе

Для каждого `input.pdf` создаётся `output.json`, дополнительно сохраняются изображения фигур и таблиц.

Пример структуры JSON:

```jsonc
{
  "metadata": {
    "title": "...",
    "title_en": null,
    "authors": ["Фамилия, И."],
    "abstract": "...",
    "keywords": [],
    "doi": "10.1039/...",
    "journal": "...",
    "year": 2023,
    "metadata_source": "crossref",
    "metadata_confidence": 0.95
  },
  "sections": [
    {
      "heading": "Introduction",
      "level": 1,
      "content": "...",
      "subsections": [],
      "number": "1"
    }
  ],
  "figures": [
    {
      "id": "Figure 1",
      "caption": "...",
      "page": 2,
      "bbox": {},
      "img_path": "images/.../fig_1_p2.png"
    }
  ],
  "tables": [
    {
      "id": "Table 1",
      "caption": "...",
      "data": [["..."]]
    }
  ],
  "equations": [
    {
      "id": "(1)",
      "latex": "E = mc^2",
      "context": "..."
    }
  ],
  "acknowledgments": null,
  "raw_text": "полный текст документа"
}
```

---

## 📈 Оценка качества

Метрики вычисляются на сравнении с эталоном `evaluation/gold.json`.

### Примеры запуска

```bash
uv run python scripts/eval_metadata.py --input ./pdfs --mode offline
uv run python -m parser --input ./pdfs --output out/
uv run python scripts/eval_pipeline.py --pred-dir out/ --gold evaluation/gold.json
```

### Ограничения

* Распознавание нестандартной нумерации формул может работать не идеально.
* Сложные таблицы зависят от качества работы VLM и могут быть частично потеряны.
* Иногда аффилиации в списке авторов вливаются в фамилии.

---

## 📝 Разработка

Перед PR убедитесь, что всё проходит:

```bash
uv run ruff check .
uv run mypy src/ tests/
uv run pytest
```
