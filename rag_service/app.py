from typing import Annotated, Final

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Query, Request, status
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig

from .database import init_database
from .exceptions import CreationError, SearchError
from .indexing import indexing_chain
from .rag import State, rag_agent, retrieve_pages
from .schemas import Message, Page, PageIndexable, Role

logging.basicConfig(level=logging.INFO)


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


@app.exception_handler(CreationError)
def handle_creation_error(request: Request, exc: CreationError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(SearchError)
def handle_search_error(request: Request, exc: SearchError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)},
    )


app.include_router(router)
