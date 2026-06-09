# Оценка качества метаданных + OCR-фоллбэк

## OCR-фоллбэк (документы без текстового слоя)

Если в PDF нет извлекаемого текста (скан), `extractor.get_page_blocks` автоматически
рендерит страницы и распознаёт их через **Tesseract** в двуязычном режиме `eng+rus`
(модуль [`parser.ocr`](../src/parser/ocr.py)). Результат — те же `PageBlock`, поэтому
метаданные/секции работают без изменений. При отсутствии Tesseract — мягкая деградация
(пустой результат + предупреждение, пайплайн не падает).

Системная зависимость:
```bash
brew install tesseract tesseract-lang      # macOS
# apt install tesseract-ocr tesseract-ocr-rus  # Linux
```

## Метрики precision / recall / F1

Эталон — `gold.json` (CrossRef по DOI + ручная проверка). Есть два режима оценки.

### 1. Только метаданные, «на лету»

Скрипт [`scripts/eval_metadata.py`](../scripts/eval_metadata.py) сам извлекает
метаданные из PDF и сравнивает по полям title / authors / doi / year / journal:
```bash
uv run python scripts/eval_metadata.py --input <папка_с_pdf> --mode offline
# режимы: offline | offline-llm | online
```

### 2. По всему документу (выход пайплайна)

Скрипт [`scripts/eval_pipeline.py`](../scripts/eval_pipeline.py) оценивает **полный
результат пайплайна** — читает готовые `Document`-JSON и считает метрики по основным
полям итогового JSON (ТЗ, доп. баллы 8.3.6): метаданные, `abstract`, `keywords`,
заголовки секций, количество фигур/таблиц/уравнений.
```bash
# 1) сначала прогоняем парсер
uv run python -m parser --input <папка_с_pdf> --output out/
# 2) затем метрики по его выходу
uv run python scripts/eval_pipeline.py --pred-dir out/ --gold evaluation/gold.json
```

Движок [`parser.evaluation`](../src/parser/evaluation.py) поддерживает три типа полей:
- **скалярные** — `title`/`journal`/`abstract` (нечёткое сравнение, вложенность строк),
  `doi`/`year` (строгое);
- **множественные** — `authors` (по фамилиям), `keywords`, `sections` (заголовки):
  per-doc P/R/F1 по множествам с macro-усреднением;
- **счётные** — `figures`/`tables`/`equations`: совпадение количества элементов.

Поле считается, только если эталон содержит для него непустое значение, поэтому
`gold.json` можно расширять постепенно (см. ниже).

### Результаты на отложенной выборке (3 статьи разных издателей)

**offline (только эвристики, без сети и LLM):**

| поле     | precision | recall | f1    |
|----------|-----------|--------|-------|
| title    | 1.000     | 1.000  | 1.000 |
| doi      | 1.000     | 1.000  | 1.000 |
| year     | 1.000     | 1.000  | 1.000 |
| authors  | 0.952     | 0.675  | 0.714 |
| journal  | 0.000     | 0.000  | 0.000 |

**online (CrossRef):** все поля 1.000 (является и эталоном, и верхней границей).

### Известные ограничения
- `journal` в оффлайн-режиме не извлекается (берётся из CrossRef онлайн).
- `authors`: высокая точность, но неполный recall на «склеенных» байлайнах
  некоторых издателей (буквы аффилиаций липнут к именам).
- Выборка мала (3 статьи с валидным DOI в CrossRef); для полноценной оценки нужно
  расширить `gold.json`.

### Что уже размечено в `gold.json` и как добавить ещё

Сейчас по всем 3 статьям размечены `title`/`authors`/`doi`/`year`/`journal`, а для 2 из
них — `abstract` (взят из CrossRef). Поля `keywords`, `sections` и счётчики
`figures`/`tables`/`equations` движок умеет считать, но они требуют **ручной**
эталонной разметки (из CrossRef не выводятся) — добавьте их в запись `gold.json`,
и метрики появятся автоматически:
```jsonc
{
  "Статья.pdf": {
    "title": "...", "authors": ["Фамилия, И."], "doi": "...",
    "keywords": ["ключевое слово", "..."],          // множество
    "sections": ["Introduction", "Methods", "..."],  // заголовки секций
    "figures": 4, "tables": 2, "equations": 5         // ожидаемые количества
  }
}
```
