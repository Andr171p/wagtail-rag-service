from langchain_core.runnables import RunnablePassthrough

from .database import bulk_add_pages
from .depends import embeddings, splitter
from .schemas import PageIndexable


def split_pages(pages: list[PageIndexable]) -> list[PageIndexable]:
    page_chunks: list[PageIndexable] = []
    for page in pages:
        content_chunks = splitter.split_text(page.content)
        page_chunks.extend(
            [
                PageIndexable(**page.model_dump(exclude={"content"}), content=content_chunk)
                for content_chunk in content_chunks
            ]
        )
    return page_chunks


async def vectorize_pages(pages: list[PageIndexable]) -> list[PageIndexable]:
    page_contents = [page.content for page in pages]
    page_embeddings = await embeddings.aembed_documents(page_contents)
    for page, embedding in zip(pages, page_embeddings, strict=False):
        page.embedding = embedding
    return pages


async def store_pages(pages: list[PageIndexable]) -> list[PageIndexable]:
    await bulk_add_pages(pages)
    return pages


indexing_chain = (
    RunnablePassthrough()
    | split_pages
    | vectorize_pages
    | store_pages
)
