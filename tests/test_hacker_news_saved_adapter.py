from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.hacker_news_saved import HackerNewsSavedAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_hacker_news_saved_imports_story_metadata(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 424242,
                    "title": "Interesting systems post",
                    "url": "https://example.com/systems",
                    "by": "pg",
                    "time": 1735689600,
                    "score": 123,
                    "type": "story",
                    "text": "A short note from the export.",
                    "kids": [1, 2, 3],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.HACKER_NEWS_SAVED
    assert unit.source_entity_type == "saved_item"
    assert unit.source_id == "hacker_news_saved:424242"
    assert unit.title == "Interesting systems post"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert "A short note from the export." in unit.content
    assert "URL: https://example.com/systems" in unit.content
    assert "Hacker News: https://news.ycombinator.com/item?id=424242" in unit.content
    assert unit.metadata["item_id"] == 424242
    assert unit.metadata["hn_item_id"] == 424242
    assert unit.metadata["author"] == "pg"
    assert unit.metadata["score"] == 123
    assert unit.metadata["item_type"] == "story"
    assert unit.metadata["hn_item_type"] == "story"
    assert unit.metadata["comment_count"] == 3
    assert unit.metadata["time"] == 1735689600
    assert unit.metadata["time_iso"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["external_url"] == "https://example.com/systems"
    assert unit.metadata["source_url"] == "https://example.com/systems"
    assert unit.metadata["hn_item_url"] == "https://news.ycombinator.com/item?id=424242"


def test_hacker_news_saved_normalizes_comment_metadata_and_references(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 424243,
                    "type": "comment",
                    "by": "dang",
                    "text": "A saved comment.",
                    "parent": 424242,
                    "story_id": 424200,
                    "time": 1735689600,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["hn_item_type"] == "comment"
    assert unit.metadata["hn_item_id"] == 424243
    assert unit.metadata["hn_parent_id"] == 424242
    assert unit.metadata["hn_story_id"] == 424200
    assert unit.tags == ["hacker_news", "comment"]


def test_hacker_news_saved_preserves_sparse_unknown_items(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps([{"id": 999, "type": "pollopt", "text": "Sparse saved item."}]),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["hn_item_id"] == 999
    assert result.units[0].metadata["hn_item_type"] == "unknown"


def test_hacker_news_saved_accepts_top_level_items_and_saved_items(tmp_path):
    items_path = tmp_path / "items.json"
    saved_items_path = tmp_path / "saved_items.json"
    items_path.write_text(json.dumps({"items": [{"id": 1, "title": "Items shape", "time": 1735689600}]}), encoding="utf-8")
    saved_items_path.write_text(
        json.dumps({"saved_items": [{"id": 2, "title": "Saved items shape", "time": 1735689601}]}),
        encoding="utf-8",
    )

    items_result = HackerNewsSavedAdapter(path=str(items_path)).ingest()
    saved_items_result = HackerNewsSavedAdapter(path=str(saved_items_path)).ingest()

    assert [unit.title for unit in items_result.units] == ["Items shape"]
    assert [unit.title for unit in saved_items_result.units] == ["Saved items shape"]


def test_hacker_news_saved_keeps_url_only_items_and_uses_hn_source_url_fallback(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {"id": 100, "url": "https://example.com/only-url", "time": 1735689600},
                {"id": 101, "title": "Ask HN item", "type": "story", "time": 1735689601},
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest()

    assert [unit.title for unit in result.units] == ["https://example.com/only-url", "Ask HN item"]
    assert result.units[0].metadata["external_url"] == "https://example.com/only-url"
    assert result.units[1].metadata["source_url"] == "https://news.ycombinator.com/item?id=101"
    assert "external_url" not in result.units[1].metadata


def test_hacker_news_saved_filters_by_sync_state_and_entity_type(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "title": "Old", "time": 1735689600},
                {"id": 2, "title": "Boundary", "time": 1735689601},
                {"id": 3, "title": "New", "time": 1735689602},
            ]
        ),
        encoding="utf-8",
    )

    skipped = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["comment"])
    result = HackerNewsSavedAdapter(path=str(path)).ingest(
        since=SyncState(
            source_project="hacker_news_saved",
            source_entity_type="saved_item",
            last_sync_at=datetime.fromtimestamp(1735689601, tz=timezone.utc),
        )
    )

    assert skipped.units == []
    assert skipped.edges == []
    assert [unit.title for unit in result.units] == ["New"]


def test_hacker_news_saved_adapter_is_registered():
    assert "hacker_news_saved" in list_adapters()
    adapter = get_adapter("hacker-news-saved", path="/tmp/saved.json")
    assert isinstance(adapter, HackerNewsSavedAdapter)
    assert adapter.name == "hacker_news_saved"
