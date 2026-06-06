import re
from typing import Any

from pydantic import BaseModel, Field, model_validator


def clean_text(text: str | None) -> str | None:
    """
    Очищает текст от trailing whitespace, висячих переносов
    и лишних разрывов строк.
    """
    if not text:
        return text

    # 1. Склеиваем переносы слов: "поверхно-\nстный" -> "поверхностный"
    cleaned = re.sub(r"-\s*\n\s*", "", text)

    # 2. Заменяем одинарные переносы строк на пробелы (сохраняя \n\n)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

    # 3. Схлопываем множественные пробелы в один
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip()


class BaseSchema(BaseModel):
    """Базовый класс с точечной автоматической очисткой строковых полей."""

    @model_validator(mode="before")
    @classmethod
    def clean_strings(cls, data: Any) -> Any:
        # Поля, в которых строжайше запрещено трогать переносы строк \n
        PROTECTED_FIELDS = {"content", "latex", "raw_text"}

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and key not in PROTECTED_FIELDS:
                    data[key] = clean_text(value)
        return data


class BBox(BaseSchema):
    """Объект с координатами (лево, верх, право, низ)."""

    left: float
    top: float
    right: float
    bottom: float

class PageBlock(BaseSchema):
    """Низкоуровневый блок контента, извлеченный со страницы PDF."""

    text: str | None = Field(
        None, description="Текст блока (null, если это картинка)"
    )
    font_size: float | None = Field(
        None, description="Максимальный размер шрифта в блоке (null для картинок)"
    )
    bbox: BBox = Field(..., description="Координаты границ блока")
    page_number: int = Field(..., description="Номер страницы (начиная с 1)")
    block_type: str = Field(..., description="Тип контента: 'text' или 'image'")
    is_bold: bool = Field(
        False, description="Признак жирного шрифта (для выделения заголовков)"
    )

class Metadata(BaseSchema):
    title: str = Field(..., description="Полное название статьи")
    title_en: str | None = Field(
        None, description="Перевод названия на английский"
    )
    authors: list[str] = Field(
        default_factory=list, description="Список авторов"
    )
    abstract: str | None = Field(
        ...,
        description="Аннотация статьи (обязательное поле, допускает null)",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Ключевые слова"
    )
    doi: str | None = Field(
        ...,
        description="DOI документа (обязательное поле, допускает null)",
    )
    journal: str | None = Field(None, description="Название журнала")
    year: int | None = Field(None, description="Год публикации")
    metadata_source: str = Field(
        ..., description="Источник: pdf, crossref, manual, hybrid"
    )
    metadata_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Оценка уверенности [0..1]"
    )


class Section(BaseSchema):
    heading: str = Field(..., description="Заголовок секции")
    level: int = Field(..., description="Уровень иерархии")
    content: str = Field(..., description="Текстовое содержимое секции")
    subsections: list["Section"] = Field(
        default_factory=list, description="Вложенные секции"
    )
    number: str | None = Field(None, description="Номер секции")


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
    panels: list[Panel] = Field(
        default_factory=list, description="Массив подпанелей"
    )


class Table(BaseSchema):
    id: str = Field(..., description="Идентификатор таблицы")
    caption: str = Field(..., description="Подпись к таблице")
    page: int = Field(..., description="Номер страницы")
    bbox: BBox = Field(..., description="Координаты таблицы")
    img_path: str = Field(..., description="Путь к изображению")
    data: list[list[str]] = Field(
        default_factory=list, description="Табличные данные"
    )


class Equation(BaseSchema):
    id: str | None = Field(None, description="Номер уравнения")
    latex: str = Field(..., description="Уравнение в формате LaTeX")
    context: str | None = Field(None, description="Окружающий контекст")
    page: int | None = Field(None, description="Номер страницы")
    bbox: BBox | None = Field(None, description="Координаты уравнения")


class Document(BaseSchema):
    metadata: Metadata = Field(..., description="Метаданные документа")
    sections: list[Section] = Field(
        default_factory=list, description="Иерархия секций"
    )
    figures: list[Figure] = Field(
        default_factory=list, description="Извлечённые фигуры"
    )
    tables: list[Table] = Field(
        default_factory=list, description="Извлечённые таблицы"
    )
    equations: list[Equation] = Field(
        default_factory=list, description="Уравнения в формате LaTeX"
    )
    acknowledgments: str | None = Field(
        None, description="Текст раздела Acknowledgements или null"
    )
    raw_text: str = Field(..., description="Полный текст документа")
