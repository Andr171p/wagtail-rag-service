from typing import Annotated, Final

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Query, status
from langchain_core.runnables import RunnableConfig

from .database import init_database
from .indexing import indexing_chain
from .rag import State, rag_agent, retrieve_pages
from .schemas import Message, Page, PageIndexable, Role


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_database()
    yield


app: Final[FastAPI] = FastAPI(
    title="API для взаимодействия с RAG по Wagtail сайту", lifespan=lifespan
)

router = APIRouter(prefix="/api/v1", tags=["REST API"])


@router.post(
    path="/pages",
    response_model=list[PageIndexable],
    status_code=status.HTTP_201_CREATED,
    summary="Добавление страницы в индекс",
)
async def create_page(page: Page) -> list[PageIndexable]:
    return await indexing_chain.ainvoke([page])


@router.get(
    path="/pages/search",
    response_model=list[Page],
    status_code=status.HTTP_200_OK,
    summary="Выполняет поиск индексированных страниц по запросу",
)
async def search_pages(
        query: Annotated[str, Query(..., description="Запрос для поиска")],
) -> list[Page]:
    state = await retrieve_pages(State(query=query))
    return [Page.model_validate(page) for page in state["pages"]]


@router.post(
    path="/rag",
    response_model=Message,
    status_code=status.HTTP_200_OK,
    summary="Отвечает на вопросы по индексированному сайту",
)
async def answer(user_message: Message) -> Message:
    config = RunnableConfig(
        configurable={"thread_id": user_message.session_id}
    )
    state = await rag_agent.ainvoke({"query": user_message.text}, config=config)
    return Message(role=Role.AI, session_id=user_message.session_id, text=state["response"])


app.include_router(router)
