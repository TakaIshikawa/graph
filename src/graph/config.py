from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "graph.db"

    forty_two_db: str = str(Path("~/Project/experiments/forty-two/forty_two.db").expanduser())
    max_db: str = str(Path("~/Project/experiments/max/max.db").expanduser())
    presence_db: str = str(Path("~/Project/experiments/presence/presence.db").expanduser())
    me_config: str = str(Path("~/Project/experiments/me/config/projects.yaml").expanduser())

    embedding_provider: str = "voyage"
    embedding_model: str = "voyage-3-lite"
    embedding_api_key: str = ""

    content_min_score: float = 7.0

    model_config = {"env_prefix": "GRAPH_"}


settings = Settings()
