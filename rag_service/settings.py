from typing import Final

from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# Лимит извлекаемых элементов по умолчанию
SEARCH_LIMIT = 10
# Лимит сообщений хранящейся в истории чата по умолчанию
MAX_CHAT_HISTORY = 10
TTL = timedelta(hours=1)
# Максимальное время ожидания в секундах ответа от LLM
LLM_TIMEOUT = 120
# Настройки для разделения страниц на чанки
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20


class GigaChatSettings(BaseSettings):
    api_key: str = ""
    scope: str = ""
    model_name: str = "GigaChat:latest"

    model_config = SettingsConfigDict(env_prefix="GIGACHAT_")


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/0"


class PostgresSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "password"
    db: str = "postgres"
    driver: str = "asyncpg"

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class EmbeddingsSettings(BaseSettings):
    base_url: str = "http://localhost:8005"
    normalize: bool = False

    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_")


class Settings(BaseSettings):
    gigachat: GigaChatSettings = GigaChatSettings()
    redis: RedisSettings = RedisSettings()
    postgres: PostgresSettings = PostgresSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()


settings: Final[Settings] = Settings()
