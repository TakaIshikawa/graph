from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.pinboard_bookmarks_json import PinboardBookmarksJsonAdapter


def test_pinboard_bookmarks_json_ingests_representative_export(tmp_path):
    export = tmp_path / "pinboard.json"
    export.write_text(
        json.dumps(
            [
                {
                    "href": "https://example.com/post",
                    "description": "Useful post",
                    "extended": "Longer note",
                    "meta": "abc",
                    "hash": "deadbeef",
                    "time": "2025-01-02T10:30:00Z",
                    "shared": "yes",
                    "toread": "no",
                    "tags": "python research",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = PinboardBookmarksJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "pinboard_bookmarks_json"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Useful post"
    assert "URL: https://example.com/post" in unit.content
    assert "Notes: Longer note" in unit.content
    assert unit.metadata["url"] == "https://example.com/post"
    assert unit.metadata["hash"] == "deadbeef"
    assert unit.metadata["shared"] is True
    assert unit.metadata["toread"] is False
    assert unit.metadata["tags"] == ["python", "research"]
    assert unit.tags == ["python", "research"]
    assert unit.created_at == datetime(2025, 1, 2, 10, 30, tzinfo=timezone.utc)


def test_pinboard_bookmarks_json_handles_object_exports_and_stable_ids(tmp_path):
    export = tmp_path / "pinboard.json"
    export.write_text(
        json.dumps({"posts": [{"href": "https://example.com/a", "description": "A", "tags": "one two"}]}),
        encoding="utf-8",
    )

    first = PinboardBookmarksJsonAdapter(path=str(export)).ingest().units[0]
    second = PinboardBookmarksJsonAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("pinboard_bookmarks_json:")
    assert first.tags == ["one", "two"]
