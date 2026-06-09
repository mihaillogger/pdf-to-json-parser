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

Скрипт [`scripts/eval_metadata.py`](../scripts/eval_metadata.py) сравнивает извлечённые
метаданные с эталоном `gold.json` (CrossRef по DOI + ручная проверка) по полям
title / authors / doi / year / journal.

Запуск:
```bash
uv run python scripts/eval_metadata.py --input <папка_с_pdf> --mode offline
# режимы: offline | offline-llm | online
```

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
