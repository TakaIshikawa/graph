from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.raindrop_bookmarks_csv import RaindropBookmarksCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_raindrop_bookmarks_csv_ingests_bookmarks_with_metadata(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,link,excerpt,note,tags,collection,created,lastUpdate,favorite,broken\n"
        "Example,https://example.com,Useful page,Remember this,\"Research, Python\",Inbox,2026-05-01T10:00:00Z,2026-05-02T11:00:00Z,true,false\n"
        "Docs,https://docs.example.com,,,\"#Docs; Reference\",Work,2026-05-03,,0,1\n",
        encoding="utf-8",
    )

    result = RaindropBookmarksCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Example", "Docs"]
    first = result.units[0]
    assert first.source_project == SourceProject.RAINDROP_BOOKMARKS_CSV
    assert first.source_entity_type == "bookmark"
    assert first.metadata["url"] == "https://example.com"
    assert first.metadata["source_url"] == "https://example.com"
    assert first.metadata["excerpt"] == "Useful page"
    assert first.metadata["note"] == "Remember this"
    assert first.metadata["tags"] == ["research", "python"]
    assert first.metadata["collection"] == "Inbox"
    assert first.metadata["favorite"] is True
    assert first.metadata["broken"] is False
    assert first.metadata["created_at"] == "2026-05-01T10:00:00+00:00"
    assert first.metadata["updated_at"] == "2026-05-02T11:00:00+00:00"
    assert first.tags == ["research", "python"]
    assert "Excerpt: Useful page" in first.content
    assert "Note: Remember this" in first.content
    assert "Collection: Inbox" in first.content
    assert result.units[1].metadata["broken"] is True
    assert result.units[1].tags == ["docs", "reference"]


def test_raindrop_bookmarks_csv_skips_blank_rows_and_handles_missing_optional_columns(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text(
        "Title,URL\n"
        ",\n"
        "Only title,\n"
        ",https://example.org\n",
        encoding="utf-8",
    )

    result = RaindropBookmarksCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Only title", "https://example.org"]
    assert result.units[0].metadata["title"] == "Only title"
    assert result.units[1].metadata["url"] == "https://example.org"


def test_raindrop_bookmarks_csv_has_stable_ids_and_since_filter(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,url,created,lastUpdate\n"
        "Old,https://old.example,2026-05-01,2026-05-01\n"
        "New,https://new.example,2026-05-03,2026-05-03\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="raindrop_bookmarks_csv",
        source_entity_type="bookmark",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    all_units = RaindropBookmarksCsvAdapter(path=str(export)).ingest().units
    filtered = RaindropBookmarksCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.title for unit in filtered] == ["New"]
    assert all_units[0].source_id == RaindropBookmarksCsvAdapter(path=str(export)).ingest().units[0].source_id


def test_raindrop_bookmarks_csv_is_registered():
    assert "raindrop_bookmarks_csv" in list_adapters()
    assert isinstance(get_adapter("raindrop-bookmarks-csv"), RaindropBookmarksCsvAdapter)
