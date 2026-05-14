from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.stackoverflow_bookmarks_json import StackOverflowBookmarksJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_stackoverflow_bookmarks_json_ingests_list_payload(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text(
        json.dumps(
            [
                {
                    "question_id": 123,
                    "title": "How do I parse CSV in Python?",
                    "link": "https://stackoverflow.com/questions/123/how-do-i-parse-csv-in-python",
                    "tags": ["python", "csv"],
                    "score": 42,
                    "answer_count": 3,
                    "accepted_answer": True,
                    "creation_date": 1735689600,
                    "bookmarked_at": "2025-01-02T03:04:05Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = StackOverflowBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.STACKOVERFLOW_BOOKMARKS_JSON
    assert unit.source_entity_type == "question_bookmark"
    assert unit.source_id == "stackoverflow_bookmarks_json:question:123"
    assert unit.title == "How do I parse CSV in Python?"
    assert unit.metadata["url"].startswith("https://stackoverflow.com/questions/123")
    assert unit.metadata["tags"] == ["python", "csv"]
    assert unit.metadata["score"] == 42
    assert unit.metadata["answer_count"] == 3
    assert unit.metadata["accepted_answer"] is True
    assert unit.metadata["creation_date"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["bookmark_date"] == "2025-01-02T03:04:05+00:00"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert "Tags: python, csv" in unit.content


def test_stackoverflow_bookmarks_json_container_directory_filters_and_malformed_files(tmp_path):
    good = tmp_path / "saved.json"
    bad = tmp_path / "bad.json"
    good.write_text(
        json.dumps(
            {
                "data": {
                    "items": [
                        {
                            "questionId": "1",
                            "questionTitle": "Old question",
                            "questionUrl": "https://stackoverflow.com/questions/1/old",
                            "tags": "python;old",
                            "bookmarkedAt": "2025-01-01",
                        },
                        {
                            "questionId": "2",
                            "questionTitle": "New question",
                            "questionUrl": "https://stackoverflow.com/questions/2/new",
                            "tagNames": "python|json",
                            "answerCount": "5",
                            "is_answered": "true",
                            "updatedAt": "2025-01-03",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    bad.write_text("{not json", encoding="utf-8")
    since = SyncState(
        source_project="stackoverflow_bookmarks_json",
        source_entity_type="question_bookmark",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = StackOverflowBookmarksJsonAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = StackOverflowBookmarksJsonAdapter(path=str(tmp_path)).ingest(entity_types=["answer"])

    assert [unit.title for unit in result.units] == ["New question"]
    assert result.units[0].metadata["tags"] == ["python", "json"]
    assert result.units[0].metadata["answer_count"] == 5
    assert result.units[0].metadata["accepted_answer"] is True
    assert skipped.units == []


def test_stackoverflow_bookmarks_json_url_fallback_id_and_registry(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text(
        json.dumps(
            {
                "bookmarks": [
                    {
                        "title": "URL only identity",
                        "url": "https://stackoverflow.com/questions/999/url-only",
                        "bookmark_date": "2025-01-01T00:00:00Z",
                    },
                    {"title": "", "url": ""},
                ]
            }
        ),
        encoding="utf-8-sig",
    )

    result = StackOverflowBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("stackoverflow_bookmarks_json:")
    assert result.units[0].source_id == StackOverflowBookmarksJsonAdapter(path=str(export)).ingest().units[0].source_id
    assert get_adapter("stackoverflow_bookmarks_json", path=str(export)).name == "stackoverflow_bookmarks_json"
