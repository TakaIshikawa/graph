from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.todoist_tasks_json import TodoistTasksJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_todoist_tasks_json_ingests_nested_tasks_due_dates_labels_and_registry(tmp_path):
    export = tmp_path / "todoist.json"
    export.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "1",
                        "content": "Ship report",
                        "description": "Draft and send",
                        "project_id": "p1",
                        "project_name": "Work",
                        "section_id": "s1",
                        "labels": ["ops", "writing"],
                        "priority": 4,
                        "due": {"date": "2026-02-01", "string": "Feb 1", "is_recurring": False},
                        "created_at": "2026-01-01T00:00:00Z",
                        "url": "https://todoist.com/showTask?id=1",
                    },
                    {
                        "id": "2",
                        "content": "Review draft",
                        "parent_id": "1",
                        "project": {"id": "p1", "name": "Work"},
                        "section": {"id": "s1", "name": "Reports"},
                        "labels": [{"name": "ops"}],
                        "completed_at": "2026-01-02T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = TodoistTasksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    parent = next(unit for unit in result.units if unit.metadata["task_id"] == "1")
    child = next(unit for unit in result.units if unit.metadata["task_id"] == "2")
    assert parent.source_project == SourceProject.TODOIST_TASKS_JSON
    assert parent.metadata["project_id"] == "p1"
    assert parent.metadata["project_name"] == "Work"
    assert parent.metadata["due"] == {"date": "2026-02-01", "string": "Feb 1", "is_recurring": False}
    assert parent.metadata["due_date"] == "2026-02-01"
    assert parent.tags == ["ops", "writing"]
    assert child.metadata["parent_id"] == "1"
    assert child.metadata["status"] == "completed"
    assert child.metadata["completed_at"] == "2026-01-02T00:00:00+00:00"
    assert get_adapter("todoist_tasks_json", path=str(export)).name == "todoist_tasks_json"


def test_todoist_tasks_json_handles_missing_optional_description_and_stable_ids(tmp_path):
    export = tmp_path / "todoist.json"
    export.write_text(json.dumps([{"id": "a", "content": "Inbox task", "created_at": "2026-01-01T00:00:00Z"}]), encoding="utf-8")

    first = TodoistTasksJsonAdapter(path=str(export)).ingest().units[0]
    second = TodoistTasksJsonAdapter(path=str(export)).ingest().units[0]

    assert first.title == "Inbox task"
    assert "description" not in first.metadata
    assert first.metadata["status"] == "active"
    assert first.source_id == second.source_id


def test_todoist_tasks_json_since_and_entity_filter(tmp_path):
    export = tmp_path / "todoist.json"
    export.write_text(
        json.dumps(
            [
                {"id": "old", "content": "Old", "created_at": "2026-01-01T00:00:00Z"},
                {"id": "new", "content": "New", "created_at": "2026-01-03T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )
    adapter = TodoistTasksJsonAdapter(path=str(export))
    sync = SyncState(source_project="todoist_tasks_json", source_entity_type="task", last_sync_at=datetime(2026, 1, 2, tzinfo=timezone.utc))

    assert [unit.title for unit in adapter.ingest(since=sync).units] == ["New"]
    assert adapter.ingest(entity_types=["project"]).units == []
