from typing import Final

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Computed, DateTime, Index, func, insert, select, text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .exceptions import CreationError, SearchError
from .schemas import PageIndexable
from .settings import SEARCH_LIMIT, settings

engine: Final[AsyncEngine] = create_async_engine(url=settings.postgres.sqlalchemy_url, echo=True)

sessionmaker: Final[async_sessionmaker[AsyncSession]] = async_sessionmaker(
        engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
)


class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(
        autoincrement=True,
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now()
    )


class PageIndexableModel(Base):
    __tablename__ = "pages"

    url: Mapped[str]
    slug: Mapped[str]
    title: Mapped[str]
    seo_metadata: Mapped[dict[str, str]] = mapped_column(JSON)
    content: Mapped[str]
    last_published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("""to_tsvector('russian',
            coalesce(title, '') || ' ' ||
            coalesce(content, '') || ' ' ||
            coalesce(seo_metadata->>'search_description', '') || ' ' ||
            coalesce(seo_metadata->>'seo_title', '') || ' ' ||
            coalesce(seo_metadata->>'meta_keywords', '') || ' ' ||
            coalesce(seo_metadata->>'tags', '')
        )"""),
    )

    __table_args__ = (
        Index("ix_pages_search_vector", "search_vector", postgresql_using="gin"),
    )


async def init_database() -> None:
    async with engine.begin() as connection:
        extensions: list[str] = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            "CREATE EXTENSION IF NOT EXISTS unaccent;",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            "CREATE EXTENSION IF NOT EXISTS btree_gin;"
        ]
        for extension in extensions:
            await connection.execute(text(extension))
        await connection.run_sync(Base.metadata.create_all)
        await connection.commit()


async def bulk_add_pages(pages: list[PageIndexable]) -> None:
    try:
        async with sessionmaker() as session:
            for page in pages:
                stmt = insert(PageIndexableModel).values(**page.model_dump())
                await session.execute(stmt)
            await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        raise CreationError(f"Error while creation pages, error: {e}") from e


async def similarity_search_pages(
        embedding: list[float], limit: int = SEARCH_LIMIT
) -> list[PageIndexable]:
    try:
        async with sessionmaker() as session:
            stmt = (
                select(PageIndexableModel)
                .order_by(PageIndexableModel.embedding.cosine_distance(embedding))
                .limit(limit)
            )
            results = await session.execute(stmt)
            models = results.scalars().all()
        return [PageIndexable.model_validate(model) for model in models]
    except SQLAlchemyError as e:
        raise SearchError(f"Error while search pages, error: {e}") from e


async def hybrid_search_pages(
        query: str, embedding: list[float], limit: int = SEARCH_LIMIT
) -> list[PageIndexable]: ...
