from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.raindrop_highlights_json import RaindropHighlightsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_raindrop_highlights_json_nested_skips_blank_filters_and_registry(tmp_path):
    path = tmp_path / "raindrop.json"
    path.write_text(json.dumps({"items": [{"text": "Old", "link": "https://e/old", "created": "2026-04-01"}, {"highlight": "New", "note": "N", "color": "yellow", "bookmark": {"title": "Article", "link": "https://e/new", "tags": ["Read"]}, "collection": {"title": "Inbox"}, "created": "2026-05-03"}, {"text": ""}]}), encoding="utf-8")
    since = SyncState(source_project="raindrop_highlights_json", source_entity_type="highlight", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = RaindropHighlightsJsonAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["Article"]
    unit = result.units[0]
    assert unit.metadata["url"] == "https://e/new"
    assert unit.metadata["collection"] == "Inbox"
    assert unit.metadata["color"] == "yellow"
    assert unit.source_id == RaindropHighlightsJsonAdapter(path=str(path)).ingest().units[1].source_id
    assert RaindropHighlightsJsonAdapter(path=str(path)).ingest(entity_types=["bookmark"]).units == []
    assert get_adapter("raindrop_highlights_json", path=str(path)).name == "raindrop_highlights_json"
