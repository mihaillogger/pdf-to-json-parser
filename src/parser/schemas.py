import re
from typing import List, Optional, Any
from pydantic import BaseModel, Field, model_validator

def clean_text(text: str | None) -> str | None:
    """Очищает текст от trailing whitespace и висячих переносов."""
    if not text:
        return text
    # Склеиваем переносы слов: "поверхно-\nстный" -> "поверхностный"
    cleaned = re.sub(r'-\s*\n\s*', '', text)
    return cleaned.strip()

class BaseSchema(BaseModel):
    """Базовый класс с автоматической очисткой строковых полей."""
    @model_validator(mode='before')
    @classmethod
    def clean_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = clean_text(value)
        return data

class BBox(BaseSchema):
    """Объект с координатами (лево, верх, право, низ)."""
    left: float
    top: float
    right: float
    bottom: float

# 4.2. Объект metadata
class Metadata(BaseSchema):
    title: str = Field(..., description="Полное название статьи")
    title_en: Optional[str] = Field(None, description="Перевод названия на английский")
    authors: List[str] = Field(default_factory=list, description="Список авторов")
    abstract: Optional[str] = Field(..., description="Аннотация статьи (обязательное поле, допускает null)")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова")
    doi: Optional[str] = Field(..., description="DOI документа (обязательное поле, допускает null)")
    journal: Optional[str] = Field(None, description="Название журнала")
    year: Optional[int] = Field(None, description="Год публикации")
    metadata_source: str = Field(..., description="Источник: pdf, crossref, manual, hybrid")
    metadata_confidence: float = Field(..., ge=0.0, le=1.0, description="Оценка уверенности [0..1]")

# 4.3. Объект Section (Рекурсивный)
class Section(BaseSchema):
    heading: str = Field(..., description="Заголовок секции")
    level: int = Field(..., description="Уровень иерархии")
    content: str = Field(..., description="Текстовое содержимое секции")
    subsections: List['Section'] = Field(default_factory=list, description="Вложенные секции")
    number: Optional[str] = Field(None, description="Номер секции")

# 4.4. Объект Figure и Panel
class Panel(BaseSchema):
    img_path: str = Field(..., description="Относительный путь к сохранённому файлу изображения")
    bbox: BBox = Field(..., description="Координаты панели")

class Figure(BaseSchema):
    id: str = Field(..., description="Идентификатор фигуры")
    caption: str = Field(..., description="Подпись к фигуре")
    page: int = Field(..., description="Номер страницы")
    bbox: BBox = Field(..., description="Координаты фигуры")
    img_path: str = Field(..., description="Путь к изображению")
    panels: List[Panel] = Field(default_factory=list, description="Массив подпанелей")

# 4.5. Объект Table
class Table(BaseSchema):
    id: str
    caption: str
    page: int
    bbox: BBox
    img_path: str
    data: List[List[str]] = Field(default_factory=list, description="Табличные данные")

# 4.6. Объект Equation
class Equation(BaseSchema):
    id: Optional[str] = Field(None, description="Номер уравнения")
    latex: str = Field(..., description="Уравнение в формате LaTeX")
    context: Optional[str] = Field(None, description="Окружающий контекст")
    page: Optional[int] = Field(None, description="Номер страницы")
    bbox: Optional[BBox] = Field(None, description="Координаты уравнения")

# 4.1. Корневая структура
class Document(BaseSchema):
    metadata: Metadata = Field(..., description="Метаданные документа")
    sections: List[Section] = Field(default_factory=list, description="Иерархия секций")
    figures: List[Figure] = Field(default_factory=list, description="Извлечённые фигуры")
    tables: List[Table] = Field(default_factory=list, description="Извлечённые таблицы")
    equations: List[Equation] = Field(default_factory=list, description="Уравнения в формате LaTeX")
    acknowledgments: Optional[str] = Field(None, description="Текст раздела Acknowledgements или null")
    raw_text: str = Field(..., description="Полный текст документа")
