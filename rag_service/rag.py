from typing import TypedDict

import logging

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .database import similarity_search_pages
from .depends import embeddings, llm, redis
from .prompts import (
    GRADE_QUERY_QUALITY_PROMPT,
    PAGE_PROMPT,
    REWRITER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from .schemas import Page, PageIndexable
from .settings import MAX_CHAT_HISTORY, TTL

logger = logging.getLogger(__name__)


class State(TypedDict):
    """FSM state of RAG agent"""
    query: str
    chat_history: list[str]
    pages: list[PageIndexable]
    response: str


def format_pages(pages: list[Page]) -> str:
    """Formating pages for LLM-friendly format"""
    return "\n\n".join([PAGE_PROMPT.format(**page.model_dump()) for page in pages])


def format_chat_history(chat_history: list[str]) -> str:
    return "\n".join(chat_history)


async def get_chat_history(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, list[str]]:
    logger.info("---RECEIVING CHAT HISTORY---")
    key = f"chat_history:{config["configurable"]["thread_id"]}"
    messages = await redis.lrange(key, 0, -1)
    return {"chat_history": [message.decode("utf-8") for message in reversed(messages)]}


async def should_rewrite_query(state: State) -> bool:

    class QueryQuality(BaseModel):
        grade: bool = Field(
            ..., description="1 если запрос качественный, 0 если запрос следует переписать"
        )

    parser = PydanticOutputParser(pydantic_object=QueryQuality)
    llm_chain = (
        ChatPromptTemplate
        .from_messages([("system", GRADE_QUERY_QUALITY_PROMPT)])
        .partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    query_quality: QueryQuality = await llm_chain.ainvoke({
        "chat_history": format_chat_history(state["chat_history"]), "query": state["query"]
    })
    return not query_quality.grade


async def rewrite_query(state: State, config: RunnableConfig | None = None) -> dict[str, str]:  # noqa: ARG001
    logger.info("---REWRITING QUERY---")
    rewriter_chain = ChatPromptTemplate.from_template(REWRITER_PROMPT) | llm | StrOutputParser()
    rewritten_query = await rewriter_chain.ainvoke({
        "chat_history": format_chat_history(state["chat_history"]), "query": state["query"]
    })
    return {"query": rewritten_query}


async def retrieve_pages(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, list[PageIndexable]]:
    logger.info("---RETRIEVE PAGES---")
    embedding = await embeddings.aembed_query(state["query"])
    pages = await similarity_search_pages(embedding)
    return {"pages": pages}


async def generate_response(state: State, config: RunnableConfig | None = None) -> dict[str, str]:  # noqa: ARG001
    logger.info("---GENERATING RESPONSE---")
    user_prompt = USER_PROMPT.format(
        chat_history=format_chat_history(state["chat_history"]), query=state["query"]
    )
    formatted_pages = format_pages(state["pages"])
    llm_chain = ChatPromptTemplate.from_template(SYSTEM_PROMPT) | llm | StrOutputParser()
    response = await llm_chain.ainvoke({"user_prompt": user_prompt, "pages": formatted_pages})
    return {"response": response}


async def persist_chat_history(state: State, config: RunnableConfig | None = None) -> State:
    logger.info("---PERSISTING CHAT HISTORY---")
    key = f"chat_history:{config["configurable"]["thread_id"]}"
    max_chat_history = config["configurable"].get("max_chat_history", MAX_CHAT_HISTORY)
    ttl = config["configurable"].get("ttl", TTL)
    user_message, ai_message = f"User: {state["query"]}", f"AI: {state["response"]}"
    await redis.lpush(key, user_message, ai_message)
    await redis.expire(key, ttl)
    await redis.ltrim(key, 0, max_chat_history - 1)
    return state


# Создание и компиляция RAG агента
workflow = StateGraph(State)
# Добавление вершин
workflow.add_node("get_chat_history", get_chat_history)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_pages", retrieve_pages)
workflow.add_node("generate_response", generate_response)
workflow.add_node("persist_chat_history", persist_chat_history)
# Добавление рёбер
workflow.add_edge(START, "get_chat_history")
workflow.add_conditional_edges(
    "get_chat_history",
    should_rewrite_query,
    {True: "rewrite_query", False: "retrieve_pages"}
)
workflow.add_edge("rewrite_query", "retrieve_pages")
workflow.add_edge("retrieve_pages", "generate_response")
workflow.add_edge("generate_response", "persist_chat_history")
workflow.add_edge("persist_chat_history", END)
# Компиляция графа
rag_agent = workflow.compile()
