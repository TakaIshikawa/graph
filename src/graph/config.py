from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    database_url: str = "graph.db"

    forty_two_db: str = ""
    max_db: str = ""
    presence_db: str = ""
    kindle_db: str = ""
    sota_db: str = ""
    feed_sources: str = ""
    bookmarks_path: str = ""
    csv_path: str = ""
    jsonl_path: str = ""
    opml_path: str = ""
    pdf_path: str = ""
    me_config: str = ""
    markdown_root: str = ""
    text_root: str = ""
    html_root: str = ""
    ical_path: str = ""
    ipynb_root: str = ""
    obsidian_vault_path: str = "/Users/taka/ObsidianVaults/note"

    embedding_provider: str = "voyage"
    embedding_model: str = "voyage-3-lite"
    embedding_api_key: str = ""

    content_min_score: float = 7.0

    model_config = {"env_prefix": "GRAPH_"}


settings = Settings()
