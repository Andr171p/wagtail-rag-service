from typing import Final

from embeddings_service.langchain import RemoteHTTPEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_gigachat import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from redis.asyncio import Redis

from .settings import CHUNK_OVERLAP, CHUNK_SIZE, LLM_TIMEOUT, settings

redis: Final[Redis] = Redis.from_url(settings.redis.url)

splitter: Final[TextSplitter] = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n#"]
)

embeddings: Final[Embeddings] = RemoteHTTPEmbeddings(
    base_url=settings.embeddings.base_url, normalize_embeddings=settings.embeddings.normalize,
)

llm: Final[BaseChatModel] = GigaChat(
    credentials=settings.gigachat.api_key,
    scope=settings.gigachat.scope,
    model=settings.gigachat.model_name,
    profanity_check=False,
    verify_ssl_certs=False,
    timeout=LLM_TIMEOUT,
)
