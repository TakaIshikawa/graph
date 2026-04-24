from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    database_url: str = "graph.db"

    embedding_provider: str = "voyage"
    embedding_model: str = "voyage-3-lite"
    embedding_api_key: str = ""

    content_min_score: float = 7.0

    model_config = {"env_prefix": "GRAPH_"}


settings = Settings()
