from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.hacker_news_submissions_csv import HackerNewsSubmissionsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_hacker_news_submissions_csv_ingests_story_metadata(tmp_path):
    export = tmp_path / "submitted.csv"
    _write_csv(
        export,
        [
            {
                "id": "424242",
                "title": "SQLite on the edge",
                "url": "https://example.com/sqlite",
                "time": "2026-05-01T10:00:00Z",
                "score": "123",
                "comments": "45",
            }
        ],
    )

    result = HackerNewsSubmissionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "hacker_news_submissions_csv"
    assert unit.source_id == "hacker_news_submissions_csv:submission:424242"
    assert unit.source_entity_type == "submission"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "SQLite on the edge"
    assert "URL: https://example.com/sqlite" in unit.content
    assert "Hacker News: https://news.ycombinator.com/item?id=424242" in unit.content
    assert unit.metadata["url"] == "https://example.com/sqlite"
    assert unit.metadata["hn_item_id"] == 424242
    assert unit.metadata["score"] == 123
    assert unit.metadata["comment_count"] == 45
    assert unit.metadata["source_file"] == "submitted.csv"
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.tags == ["hacker_news", "submitted", "story"]


def test_hacker_news_submissions_csv_falls_back_to_hn_url_for_text_submission(tmp_path):
    export = tmp_path / "submitted.csv"
    _write_csv(export, [{"Item ID": "7", "Title": "Ask HN: What are you reading?", "Text": "Books?", "Comments": ""}])

    unit = HackerNewsSubmissionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["url"] == "https://news.ycombinator.com/item?id=7"
    assert "Hacker News: https://news.ycombinator.com/item?id=7" in unit.content
    assert "comment_count" not in unit.metadata
    assert unit.tags == ["hacker_news", "submitted", "text"]


def test_hacker_news_submissions_csv_stable_ids_and_missing_optional_metrics(tmp_path):
    export = tmp_path / "submitted.csv"
    _write_csv(
        export,
        [
            {"Title": "No metrics", "URL": "https://example.com/a"},
            {"Title": "No metrics", "URL": "https://example.com/a"},
        ],
    )

    result = HackerNewsSubmissionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("hacker_news_submissions_csv:submission:")
    assert "score" not in result.units[0].metadata
    assert "comment_count" not in result.units[0].metadata


def test_hacker_news_submissions_csv_filters_since_and_entity_type(tmp_path):
    export = tmp_path / "submitted.csv"
    _write_csv(
        export,
        [
            {"Title": "Old", "Item ID": "1", "Submitted At": "2026-05-01T00:00:00Z"},
            {"Title": "New", "Item ID": "2", "Submitted At": "2026-05-03T00:00:00Z"},
        ],
    )
    since = SyncState(
        source_project="hacker_news_submissions_csv",
        source_entity_type="submission",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    skipped = HackerNewsSubmissionsCsvAdapter(path=str(export)).ingest(entity_types=["other"])
    result = HackerNewsSubmissionsCsvAdapter(path=str(export)).ingest(since=since)

    assert skipped.units == []
    assert [unit.title for unit in result.units] == ["New"]
