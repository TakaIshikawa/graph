from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_search_history_json import GoogleSearchHistoryJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_google_search_history_json_ingests_takeout_query_metadata_and_registry(tmp_path):
    export = tmp_path / "search.json"
    export.write_text(
        json.dumps(
            {
                "activity": [
                    {
                        "title": "Searched for graph databases",
                        "titleUrl": "https://www.google.com/search?q=graph%20databases",
                        "time": "2025-01-02T03:04:05Z",
                        "products": ["Search"],
                        "details": [{"name": "From Chrome"}],
                        "device": "Mac",
                        "application": "Chrome",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GoogleSearchHistoryJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_SEARCH_HISTORY_JSON
    assert unit.source_entity_type == "search_query"
    assert unit.metadata["query"] == "graph databases"
    assert unit.metadata["url"] == "https://www.google.com/search?q=graph%20databases"
    assert unit.metadata["products"] == ["Search"]
    assert unit.metadata["details"] == ["From Chrome"]
    assert unit.metadata["device"] == "Mac"
    assert unit.metadata["application"] == "Chrome"
    assert unit.metadata["source_file"] == "search.json"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("google_search_history_json", path=str(export)).name == "google_search_history_json"


def test_google_search_history_json_explicit_query_time_usec_since_bad_files_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"query": "old", "time": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"items": [{"query": "new", "url": "https://google.com/search?q=new", "time_usec": "1735862400000000"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = GoogleSearchHistoryJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="google_search_history_json", source_entity_type="search_query", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["new"]
    assert first.units[0].updated_at == datetime(2025, 1, 3, tzinfo=timezone.utc)
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["activity"]).units == []
