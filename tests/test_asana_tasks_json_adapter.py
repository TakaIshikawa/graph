import json
from datetime import datetime, timezone

from graph.adapters.asana_tasks_json import AsanaTasksJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_asana_tasks_json_ingests_wrapped_data_and_nested_metadata(tmp_path):
    path = tmp_path / "asana.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "gid": "120",
                        "name": "Write adapter",
                        "notes": "Parse API payloads",
                        "permalink_url": "https://app.asana.com/0/1/120",
                        "completed": False,
                        "due_on": "2026-06-10",
                        "start_on": "2026-06-01",
                        "projects": [{"gid": "p1", "name": "Graph"}],
                        "workspace": {"gid": "w1", "name": "Acme"},
                        "assignee": {"gid": "u1", "name": "Ada"},
                        "tags": [{"name": "backend"}, {"name": "json"}],
                        "created_at": "2026-05-01T10:00:00Z",
                        "modified_at": "2026-05-02T10:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = AsanaTasksJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "asana_tasks_json:120"
    assert unit.metadata["completed"] is False
    assert unit.metadata["due_on"] == "2026-06-10"
    assert unit.metadata["start_on"] == "2026-06-01"
    assert unit.metadata["projects"] == ["Graph"]
    assert unit.metadata["workspace"] == "Acme"
    assert unit.metadata["assignee"] == "Ada"
    assert unit.metadata["tags"] == ["backend", "json"]
    assert {"asana", "task", "Graph", "backend", "json"}.issubset(set(unit.tags))


def test_asana_tasks_json_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"gid": "old", "name": "Old", "modified_at": "2026-04-01T00:00:00Z"},
                    {"gid": "new", "name": "New", "completed": True, "completed_at": "2026-05-03T00:00:00Z", "modified_at": "2026-05-02T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(source_project="asana_tasks_json", source_entity_type="task", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = AsanaTasksJsonAdapter(str(path)).ingest(since=since, entity_types=["task"])

    assert [unit.source_id for unit in result.units] == ["asana_tasks_json:new"]
    assert result.units[0].metadata["status"] == "completed"
    assert result.units[0].metadata["completed"] is True
    assert AsanaTasksJsonAdapter(str(path)).ingest(entity_types=["project"]).units == []
    assert get_adapter("asana_tasks_json", path=str(path)).name == "asana_tasks_json"


def test_asana_tasks_json_flat_items_malformed_and_fallback_ids(tmp_path):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text(json.dumps({"items": [{"name": "No gid", "notes": "Digest me"}, {}]}), encoding="utf-8")
    invalid.write_text("{", encoding="utf-8")

    result = AsanaTasksJsonAdapter(str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("asana_tasks_json:")
    assert result.units[0].title == "No gid"
