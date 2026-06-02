import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.slack_bookmarks_json import SlackBookmarksJsonAdapter
from graph.types.models import SyncState


def test_slack_bookmarks_json_ingests_channel_scoped_and_flat_payloads(tmp_path):
    path = tmp_path / "bookmarks.json"
    path.write_text(json.dumps({"channels": [{"id": "C1", "name": "eng", "bookmarks": [{"id": "B1", "title": "Runbook", "link": "https://example.com/runbook", "user_id": "U1", "date_created": "2026-05-01T00:00:00Z"}]}], "items": [{"id": "B2", "title": "Flat", "url": "https://example.com/flat", "channel_name": "ops", "date_updated": "2026-05-03T00:00:00Z"}]}), encoding="utf-8")

    units = SlackBookmarksJsonAdapter(str(path)).ingest().units

    assert [unit.source_id for unit in units] == ["slack_bookmarks_json:B1", "slack_bookmarks_json:B2"]
    assert units[0].metadata["channel"] == "eng"
    assert units[0].metadata["channel_id"] == "C1"
    assert units[0].metadata["link"] == "https://example.com/runbook"
    assert {"slack", "bookmark", "eng"}.issubset(set(units[0].tags))


def test_slack_bookmarks_json_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "bookmarks.json"
    path.write_text(json.dumps({"bookmarks": [{"id": "old", "title": "Old", "date_updated": "2026-04-01T00:00:00Z"}, {"id": "new", "title": "New", "date_created": "2026-05-02T00:00:00Z"}]}), encoding="utf-8")
    since = SyncState(source_project="slack_bookmarks_json", source_entity_type="bookmark", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = SlackBookmarksJsonAdapter(str(path)).ingest(since=since, entity_types=["bookmark"])

    assert [unit.source_id for unit in result.units] == ["slack_bookmarks_json:new"]
    assert SlackBookmarksJsonAdapter(str(path)).ingest(entity_types=["message"]).units == []
    assert get_adapter("slack_bookmarks_json", path=str(path)).name == "slack_bookmarks_json"
