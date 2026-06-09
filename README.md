# 📄 PDF → JSON: Парсер научных статей по химии

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

CLI-инструмент и Web-интерфейс для пакетного извлечения структурированных данных из научных PDF-статей в строгий машиночитаемый **JSON по схеме ТЗ**. Рассчитан на батч-обработку ~100 документов, кросс-платформенный (Linux / macOS / Windows / WSL).

Парсер вытаскивает из каждого PDF:
* **Метаданные** — заголовок, авторы, DOI, год, журнал, аннотация, ключевые слова.
* **Секции** — иерархическое дерево разделов с заголовками и текстом.
* **Фигуры** — координаты, кропы изображений, подписи (caption) и ID.
* **Таблицы** — кропы, подписи и распознанные данные (2D-массив).
* **Уравнения** — LaTeX, номер, координаты и текстовый контекст.
* **Полный текст** документа.

---

## ✨ Возможности

* 🌍 **Три режима работы**: Онлайн (CrossRef), Офлайн + локальная LLM, Офлайн без LLM.
* 🔒 **Оффлайн по умолчанию**: Внешние API (CrossRef/OpenAlex) и LLM строго опциональны и отключаются флагами. Ваши данные не покидают машину.
* 📝 **OCR-фоллбэк**: Поддержка сканов без текстового слоя (Tesseract, `eng+rus`).
* ⚡ **Параллельная батч-обработка**: Ускорение парсинга через `--workers`.
* 📊 **Детальное логирование**: Статус по каждому документу + итоговая сводка в `run.log`.
* 🛡️ **Строгие схемы Pydantic**: Гарантированно валидный JSON на выходе.
* 📈 **Встроенные метрики**: Оценка Precision / Recall / F1 на отложенной выборке.

---

## 🏗 Архитектура

Модульная архитектура, где каждый слой тестируется независимо. Оркестратор собирает результаты в объект `Document` и сериализует в JSON.

| Модуль | Назначение |
| :--- | :--- |
| `cli.py` | CLI на Typer: парсинг аргументов, режимы, логирование |
| `pipeline.py` | Оркестратор: полный цикл обработки + статусы/сводка |
| `schemas.py` | Pydantic-схемы итогового JSON (стандарт данных) |
| `extractor.py` | Извлечение текстовых блоков с учётом колонок (PyMuPDF) |
| `ocr.py` | OCR-фоллбэк для сканов (Tesseract) |
| `metadata.py` | Каскад метаданных: DOI → CrossRef → LLM → эвристики |
| `sections.py` | Построение иерархического дерева секций |
| `figures.py` | Фигуры/таблицы (DocLayout-YOLO + VLM для таблиц) |
| `equations.py` | Уравнения (YOLO + Pix2Tex → LaTeX) |
| `evaluation.py` | Расчет метрик качества |

---

## 🚀 Быстрый старт (Docker + Streamlit)

Самый надежный способ запуска, не требующий установки системных зависимостей на хост-машину.

### 1. Подготовка Ollama (Для метаданных и таблиц)
Парсер использует локальные нейросети. Установите [Ollama](https://ollama.com) на ваш компьютер и выкачайте модели:
```bash
ollama run qwen2.5:3b  # Для добора метаданных
ollama run llava       # Для распознавания таблиц

```

### 2. Запуск контейнера

Склонируйте репозиторий и запустите сборку (первый запуск скачает необходимые ML-библиотеки в кэш):

```bash
git clone [https://github.com/mihaillogger/pdf-to-json-parser.git](https://github.com/mihaillogger/pdf-to-json-parser.git)
cd pdf-to-json-parser

# Запуск в фоновом режиме
docker-compose up --build -d

```

> Web-интерфейс будет доступен по адресу: **http://localhost:8501**

Контейнер автоматически пробросит запросы к вашей локальной Ollama через `host.docker.internal`.

---

## 💻 Локальная установка (CLI)

Если вы хотите запускать скрипты напрямую:

### Требования

* **Python 3.10–3.12** (на 3.13+ часть ML-зависимостей не имеет колёс).
* **[uv](https://docs.astral.sh/uv/)** — быстрый менеджер зависимостей.
* Системные пакеты (Tesseract, Poppler, Ghostscript, libgl1):
```bash
# Debian/Ubuntu / WSL
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus ghostscript poppler-utils libgl1 libglib2.0-0

# macOS
brew install tesseract tesseract-lang ghostscript poppler

```



### Инициализация

```bash
uv sync  # Поднимет .venv и поставит зависимости строго по uv.lock

```

### Использование CLI

```bash
# Обработка одного файла
uv run python -m parser --input article.pdf --output out/

# Батч-обработка директории (4 воркера)
uv run python -m parser --input ./pdfs --output out/ --workers 4

```

### Флаги CLI

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--workers` | `1` | Число параллельных процессов |
| `--offline` | `off` | Принудительный оффлайн режим (без сети) |
| `--no-crossref` | `off` | Отключить CrossRef API |
| `--no-llm` | `off` | Отключить локальную LLM (Ollama) |
| `--extract-images` | `on` | Извлекать фигуры и таблицы (YOLO + VLM) |

---

## 📊 Формат вывода

На каждый `<имя>.pdf` создаётся `<имя>.json` (объект `Document`), кропы фигур/таблиц в `out/images/<имя>/`, а также общий `out/run.log`.

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

## 📈 Метрики качества

Метрики (Precision, Recall, F1) рассчитываются на отложенной выборке (held-out set) относительно эталонной разметки `evaluation/gold.json`. Эталон формируется из независимого источника (CrossRef) и ручной валидации.

### Запуск метрик

```bash
# Оценка метаданных "на лету"
uv run python scripts/eval_metadata.py --input ./pdfs --mode offline

# Оценка полного пайплайна
uv run python -m parser --input ./pdfs --output out/
uv run python scripts/eval_pipeline.py --pred-dir out/ --gold evaluation/gold.json

```

### Краткий отчёт (Offline режим)

*Без использования сети и LLM (только базовые эвристики):*

| Поле | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **Title** | 1.000 | 1.000 | 1.000 |
| **DOI** | 1.000 | 1.000 | 1.000 |
| **Year** | 1.000 | 1.000 | 1.000 |
| **Authors** | 0.952 | 0.675 | 0.714 |
| **Abstract** | 1.000 | 0.500 | 0.667 |

> В режиме **Online** (CrossRef) качество извлечения по всем метаданным достигает **1.000** (верхняя граница).

---

## 🛠 Известные ограничения (Edge Cases)

* **Нумерация формул:** Кастомная YOLOv8 уверенно детектирует классический формат `(1)`. Специфическая вёрстка (например, `(eqn 1)`) может быть пропущена.
* **Извлечение таблиц:** Конвертация сложных таблиц в 2D-массив (`data`) зависит от качества работы VLM (`llava`) и вычислительных мощностей (GPU). В случае неудачи сохраняется BBox и оригинальное изображение таблицы.
* **Имена авторов:** В байлайнах некоторых издателей буквы аффилиаций (сноски) могут "прилипать" к фамилиям, снижая общий Recall.

---

## 👥 Команда

| Участник | Зона ответственности |
| --- | --- |
| **Матвей Ильенков** | `extractor.py`: Извлечение текста, обработка колонок, зонирование |
| **Роман Корняков** | `sections.py`: Дерево секций, Docker-инфраструктура, Web-интерфейс |
| **Арсений Фёдоров** | `figures.py`, `equations.py`: Детекция фигур/таблиц/уравнений, скрипты оценки качества |
| **Михаил Позин** | `pipeline.py`, `cli.py`: Архитектура пайплайна, CLI, интеграция CI/CD |
| **Арсений Бобченок** | `metadata.py`, `ocr.py`: Каскад DOI/CrossRef/LLM, OCR-фоллбэк |

---

## 📝 Разработка

Перед созданием Pull Request убедитесь, что код проходит все проверки:

```bash
uv run ruff check .          # Линтер
uv run mypy src/ tests/      # Типизация (strict)
uv run pytest                # Unit-тесты

```