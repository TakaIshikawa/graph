from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.linear_documents_json import LinearDocumentsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_linear_documents_json_data_documents_markdown_archived_filters_and_registry(tmp_path):
    path = tmp_path / "linear.json"
    path.write_text(json.dumps({"data": {"documents": [{"id": "old", "title": "Old", "content": "old", "updatedAt": "2026-04-01"}, {"id": "doc1", "title": "Spec", "content": "# Heading\nBody", "creator": {"name": "Ann"}, "project": {"name": "Proj"}, "team": {"key": "ENG"}, "archived": True, "url": "https://linear/doc1", "updatedAt": "2026-05-03"}]}}), encoding="utf-8")
    since = SyncState(source_project="linear_documents_json", source_entity_type="document", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = LinearDocumentsJsonAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["Spec"]
    unit = result.units[0]
    assert unit.source_id == "linear_documents_json:doc1"
    assert "# Heading\nBody" in unit.content
    assert unit.metadata["archived"] is True
    assert unit.metadata["project"] == "Proj"
    assert LinearDocumentsJsonAdapter(path=str(path)).ingest(entity_types=["issue"]).units == []
    assert get_adapter("linear_documents_json", path=str(path)).name == "linear_documents_json"
