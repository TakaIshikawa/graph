from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.hacker_news_upvoted_csv import HackerNewsUpvotedCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_hacker_news_upvoted_csv_ingests_item_metadata(tmp_path):
    export = tmp_path / "upvoted.csv"
    _write_csv(
        export,
        [
            {
                "Title": "SQLite on the edge",
                "URL": "https://example.com/sqlite",
                "Item ID": "424242",
                "Type": "story",
                "Author": "pg",
                "Score": "123",
                "Comments": "45",
                "Created At": "2026-05-01T10:00:00Z",
                "Upvoted At": "2026-05-02T12:30:00Z",
            }
        ],
    )

    result = HackerNewsUpvotedCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "hacker_news_upvoted_csv"
    assert unit.source_id == "hacker_news_upvoted_csv:upvoted_item:424242"
    assert unit.source_entity_type == "upvoted_item"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "SQLite on the edge"
    assert "URL: https://example.com/sqlite" in unit.content
    assert "Hacker News: https://news.ycombinator.com/item?id=424242" in unit.content
    assert unit.metadata["url"] == "https://example.com/sqlite"
    assert unit.metadata["item_id"] == 424242
    assert unit.metadata["hn_item_id"] == 424242
    assert unit.metadata["author"] == "pg"
    assert unit.metadata["score"] == 123
    assert unit.metadata["comment_count"] == 45
    assert unit.metadata["source_file"] == "upvoted.csv"
    assert unit.metadata["source_row"]["Title"] == "SQLite on the edge"
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)
    assert unit.tags == ["hacker_news", "story"]


def test_hacker_news_upvoted_csv_directory_skips_bad_files_dedupes_and_sorts(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    bad = tmp_path / "bad.csv"
    _write_csv(
        first,
        [
            {"Title": "Second", "Item ID": "2", "Upvoted At": "2026-05-02T00:00:00Z"},
            {"Title": "First", "Item ID": "1", "Upvoted At": "2026-05-01T00:00:00Z"},
        ],
    )
    _write_csv(second, [{"Title": "Second duplicate", "Item ID": "2", "Upvoted At": "2026-05-03T00:00:00Z"}])
    bad.write_bytes(b"\xff\xfe\x00")

    result = HackerNewsUpvotedCsvAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "hacker_news_upvoted_csv:upvoted_item:1",
        "hacker_news_upvoted_csv:upvoted_item:2",
    ]
    assert [unit.title for unit in result.units] == ["First", "Second duplicate"]
    assert [(unit.updated_at, unit.source_id) for unit in result.units] == sorted((unit.updated_at, unit.source_id) for unit in result.units)


def test_hacker_news_upvoted_csv_filters_since_and_entity_type(tmp_path):
    export = tmp_path / "upvoted.csv"
    _write_csv(
        export,
        [
            {"Title": "Old", "Item ID": "1", "Upvoted At": "2026-05-01T00:00:00Z"},
            {"Title": "Boundary", "Item ID": "2", "Upvoted At": "2026-05-02T00:00:00Z"},
            {"Title": "New", "Item ID": "3", "Upvoted At": "2026-05-03T00:00:00Z"},
        ],
    )
    since = SyncState(
        source_project="hacker_news_upvoted_csv",
        source_entity_type="upvoted_item",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    skipped = HackerNewsUpvotedCsvAdapter(path=str(export)).ingest(entity_types=["saved_item"])
    result = HackerNewsUpvotedCsvAdapter(path=str(export)).ingest(since=since)

    assert skipped.units == []
    assert [unit.title for unit in result.units] == ["New"]
