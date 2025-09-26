from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SEOMetadata(BaseModel):
    """SEO метаданные Wagtail страницы"""
    seo_title: str
    tags: list[str] = Field(default_factory=list)
    search_description: str
    meta_keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel):
    """Wagtail страница для RAG индексации.

    Attributes:
        id: Уникальный идентификатор страницы в Wagtail.
        url: Полный URL адрес страницы.
        slug: Человеко-читаемый tag в URL адресе.
        title: Заголовок страницы.
        seo_metadata: Метаданные для SEO продвижения.
        content: Основное текстовый контент страницы в формате Markdown.
        last_published_at: Дата последней публикации страницы.
    """
    id: int
    url: str
    slug: str
    title: str
    seo_metadata: SEOMetadata
    content: str
    last_published_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PageIndexable(Page):
    """Индексируемая страница"""
    embedding: list[float] = Field(default_factory=list)


class Role(StrEnum):
    """Роль сообщения"""
    USER = "user"
    AI = "ai"


class Message(BaseModel):
    """Сообщение для диалога AI агентом"""
    role: Role
    session_id: str
    text: str
