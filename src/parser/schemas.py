import re
from typing import Any

from pydantic import BaseModel, Field, model_validator


def clean_text(text: str | None) -> str | None:
    """
    Очищает текст от trailing whitespace, висячих переносов
    и артефактов вёрстки (неразрывных пробелов, лишних переносов строк).
    """
    if not text:
        return text

    # 1. Заменяем неразрывный пробел \xa0 на обычный
    cleaned = text.replace("\xa0", " ")

    # 2. Склеиваем переносы слов: "поверхно-\nстный" -> "поверхностный"
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)

    # 3. Заменяем одинарные переносы строк на пробелы (сохраняя \n\n)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

    # 4. Схлопываем множественные пробелы в один, которые могли появиться
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip()


class BaseSchema(BaseModel):
    """Базовый класс с автоматической очисткой строковых полей."""

    @model_validator(mode="before")
    @classmethod
    def clean_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ("latex", "raw_text", "content"):
                    continue

                if isinstance(value, str):
                    data[key] = clean_text(value)
        return data


class BBox(BaseSchema):
    """Объект с координатами (лево, верх, право, низ)."""

    left: float
    top: float
    right: float
    bottom: float


class PageBlock(BaseSchema):
    """Низкоуровневый блок контента, извлеченный со страницы PDF.

    Используется как промежуточный формат для передачи сырых данных
    от I/O экстрактора в алгоритмы сборки секций и фигур.
    """

    text: str | None = Field(None, description="Текст блока (null, если это картинка)")
    font_size: float | None = Field(
        None, description="Максимальный размер шрифта в блоке (null для картинок)"
    )
    bbox: BBox = Field(..., description="Координаты границ блока")
    page_number: int = Field(..., description="Номер страницы (начиная с 1)")
    block_type: str = Field(..., description="Тип контента: 'text' или 'image'")
    is_bold: bool = Field(False, description="Признак жирного шрифта (для заголовков)")


# 4.2. Объект metadata


class Metadata(BaseSchema):
    title: str = Field(..., description="Полное название статьи")
    title_en: str | None = Field(None, description="Перевод названия на английский")
    authors: list[str] = Field(default_factory=list, description="Список авторов")
    abstract: str | None = Field(
        ..., description="Аннотация статьи (обязательное поле, допускает null)"
    )
    keywords: list[str] = Field(default_factory=list, description="Ключевые слова")
    doi: str | None = Field(
        ..., description="DOI документа (обязательное поле, допускает null)"
    )
    journal: str | None = Field(None, description="Название журнала")
    year: int | None = Field(None, description="Год публикации")
    metadata_source: str = Field(
        ..., description="Источник: pdf, crossref, manual, hybrid"
    )
    metadata_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Оценка уверенности [0..1]"
    )
    normative: Any | None = Field(
        None, description="Поле из эталона, по умолчанию null"
    )


# 4.3. Объект Section (Рекурсивный)


class Section(BaseSchema):
    heading: str = Field(..., description="Заголовок секции")
    level: int = Field(..., description="Уровень иерархии")
    content: str = Field(..., description="Текстовое содержимое секции")
    subsections: list["Section"] = Field(
        default_factory=list, description="Вложенные секции"
    )
    number: str | None = Field(None, description="Номер секции")
    status: str | None = Field(None, description="Статус")
    status_effective_from: str | None = Field(
        None, description="Дата вступления статуса"
    )


# 4.4. Объект Figure и Panel


class Panel(BaseSchema):
    img_path: str = Field(
        ..., description="Относительный путь к сохранённому файлу изображения"
    )
    bbox: BBox = Field(..., description="Координаты панели")


class Figure(BaseSchema):
    id: str = Field(..., description="Идентификатор фигуры")
    caption: str = Field(..., description="Подпись к фигуре")
    page: int = Field(..., description="Номер страницы")
    bbox: BBox = Field(..., description="Координаты фигуры")
    img_path: str = Field(..., description="Путь к изображению")
    panels: list[Panel] = Field(default_factory=list, description="Массив подпанелей")


# 4.5. Объект Table


class TableDataResponse(BaseModel):
    """Вспомогательная схема для валидации JSON-ответа от VLM (LLaVA)."""
    data: list[list[str]] = Field(..., description="Двумерный массив табличных данных")


class Table(BaseSchema):
    id: str
    caption: str
    page: int
    bbox: BBox
    img_path: str
    data: list[list[str]] = Field(default_factory=list, description="Табличные данные")


# 4.6. Объект Equation


class Equation(BaseSchema):
    id: str | None = Field(None, description="Номер уравнения")
    latex: str = Field(..., description="Уравнение в формате LaTeX")
    context: str | None = Field(None, description="Окружающий контекст")
    page: int | None = Field(None, description="Номер страницы")
    bbox: BBox | None = Field(None, description="Координаты уравнения")


# 4.1. Корневая структура


class Document(BaseSchema):
    metadata: Metadata = Field(..., description="Метаданные документа")
    sections: list[Section] = Field(default_factory=list, description="Иерархия секций")
    figures: list[Figure] = Field(
        default_factory=list, description="Извлечённые фигуры"
    )
    tables: list[Table] = Field(default_factory=list, description="Извлечённые таблицы")
    equations: list[Equation] = Field(
        default_factory=list, description="Уравнения в формате LaTeX"
    )
    acknowledgments: str | None = Field(
        None, description="Текст раздела Acknowledgements или null"
    )
    raw_text: str = Field(..., description="Полный текст документа")
