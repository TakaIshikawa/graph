from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.chrome_reading_list_json import ChromeReadingListJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_chrome_reading_list_json_ingests_list_payload(tmp_path):
    export = tmp_path / "reading-list.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "Example Article",
                    "url": "HTTPS://Example.com/article/",
                    "addedAt": "2026-05-01T12:00:00Z",
                    "updatedAt": "2026-05-02T12:00:00Z",
                    "read": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = ChromeReadingListJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.CHROME_READING_LIST_JSON
    assert unit.source_entity_type == "reading_list_item"
    assert unit.title == "Example Article"
    assert unit.metadata["url"] == "HTTPS://Example.com/article/"
    assert unit.metadata["normalized_url"] == "https://example.com/article"
    assert unit.metadata["domain"] == "example.com"
    assert unit.metadata["read"] is True
    assert unit.metadata["added_at"] == "2026-05-01T12:00:00+00:00"
    assert unit.updated_at == datetime(2026, 5, 2, 12, tzinfo=timezone.utc)
    assert "URL: HTTPS://Example.com/article/" in unit.content
    assert unit.source_id == ChromeReadingListJsonAdapter(path=str(export)).ingest().units[0].source_id


def test_chrome_reading_list_json_directory_filters_duplicates_and_malformed_files(tmp_path):
    (tmp_path / "one.json").write_text(
        json.dumps(
            {
                "data": {
                    "items": [
                        {"name": "Old", "href": "https://example.com/old", "timeAdded": "2026-04-30"},
                        {"name": "New", "href": "https://example.com/new", "lastUpdated": "2026-05-03", "readStatus": "unread"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "two.json").write_text(
        json.dumps({"readingList": [{"title": "Duplicate URL", "url": "https://example.com/new/", "updatedAt": "2026-05-04"}]}),
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    since = SyncState(
        source_project="chrome_reading_list_json",
        source_entity_type="reading_list_item",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = ChromeReadingListJsonAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = ChromeReadingListJsonAdapter(path=str(tmp_path)).ingest(entity_types=["bookmark"])

    assert [unit.title for unit in result.units] == ["New", "Duplicate URL"]
    assert result.units[0].metadata["read"] is False
    assert result.units[0].source_id == result.units[1].source_id
    assert skipped.units == []
    assert get_adapter("chrome_reading_list_json", path=str(tmp_path)).name == "chrome_reading_list_json"


def test_chrome_reading_list_json_skips_empty_records(tmp_path):
    export = tmp_path / "reading-list.json"
    export.write_text(json.dumps({"items": [{"title": "", "url": ""}, {"url": "https://example.com"}]}), encoding="utf-8-sig")

    result = ChromeReadingListJsonAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["https://example.com"]
