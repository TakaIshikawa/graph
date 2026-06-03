from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_drive_comments_json import GoogleDriveCommentsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_google_drive_comments_json_grouped_replies_resolved_filters_and_registry(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(json.dumps({"file": {"id": "f1", "title": "Doc", "url": "https://docs/f1"}, "comments": [{"id": "old", "content": "Old", "createdTime": "2026-04-01"}, {"id": "new", "content": "New", "quotedText": "Quote", "author": {"displayName": "Ann"}, "resolved": True, "createdTime": "2026-05-03", "replies": [{"content": "Reply", "author": {"displayName": "Ben"}}]}, {"id": "blank"}]}), encoding="utf-8")
    since = SyncState(source_project="google_drive_comments_json", source_entity_type="comment", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = GoogleDriveCommentsJsonAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["Doc"]
    unit = result.units[0]
    assert unit.metadata["file_id"] == "f1"
    assert unit.metadata["resolved"] is True
    assert unit.metadata["replies"][0]["text"] == "Reply"
    assert GoogleDriveCommentsJsonAdapter(path=str(path)).ingest(entity_types=["file"]).units == []
    assert get_adapter("google_drive_comments_json", path=str(path)).name == "google_drive_comments_json"
