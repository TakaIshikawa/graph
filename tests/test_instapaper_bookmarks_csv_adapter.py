from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.instapaper_bookmarks_csv import InstapaperBookmarksCsvAdapter
from graph.types.models import SyncState


def test_instapaper_bookmarks_csv_ingests_saved_articles_with_folder_and_progress(tmp_path):
    export = tmp_path / "bookmarks.csv"
    export.write_text(
        "Title,URL,Folder,Selection,Description,Progress,Added\n"
        "Essay,https://example.com/essay,Read Later,Important bit,An essay,42%,2026-05-01\n"
        ",https://example.com/url-only,,,,,2026-05-02\n",
        encoding="utf-8",
    )

    result = InstapaperBookmarksCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["saved_article", "saved_article"]
    unit = result.units[0]
    assert unit.metadata["folder"] == "Read Later"
    assert unit.metadata["progress"] == 42.0
    assert "Folder: Read Later" in unit.content
    assert "Progress: 42" in unit.content
    assert "Selection: Important bit" in unit.content


def test_instapaper_bookmarks_csv_url_source_ids_are_order_stable_and_filters(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "Title,URL,Added\n"
        "A,https://example.com/a,2026-05-01\n"
        "B,https://example.com/b,2026-05-03\n",
        encoding="utf-8",
    )
    second.write_text(
        "Title,URL,Added\n"
        "B,https://example.com/b,2026-05-03\n"
        "A,https://example.com/a,2026-05-01\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="instapaper_bookmarks_csv",
        source_entity_type="saved_article",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    first_ids = sorted(unit.source_id for unit in InstapaperBookmarksCsvAdapter(path=str(first)).ingest().units)
    second_ids = sorted(unit.source_id for unit in InstapaperBookmarksCsvAdapter(path=str(second)).ingest().units)

    assert first_ids == second_ids
    assert [unit.metadata["title"] for unit in InstapaperBookmarksCsvAdapter(path=str(first)).ingest(since=since).units] == ["B"]
    assert InstapaperBookmarksCsvAdapter(path=str(first)).ingest(entity_types=["highlight"]).units == []
