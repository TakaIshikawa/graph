from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_tasks_json import GoogleTasksJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_google_tasks_json_ingests_nested_completed_due_and_links(tmp_path):
    export = tmp_path / "tasks.json"
    export.write_text(
        json.dumps(
            {
                "task_lists": [
                    {
                        "title": "Work",
                        "tasks": [
                            {
                                "id": "parent",
                                "title": "Parent task",
                                "notes": "See docs",
                                "status": "needsAction",
                                "due": "2025-01-05T00:00:00Z",
                                "updated": "2025-01-02T00:00:00Z",
                                "links": [{"type": "email", "link": "https://example.com"}],
                                "children": [
                                    {
                                        "id": "child",
                                        "title": "Child task",
                                        "status": "completed",
                                        "completed": "2025-01-03T00:00:00Z",
                                        "updated": "2025-01-03T00:00:00Z",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GoogleTasksJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["google_tasks_json:child", "google_tasks_json:parent"]
    parent = next(unit for unit in result.units if unit.metadata["task_id"] == "parent")
    child = next(unit for unit in result.units if unit.metadata["task_id"] == "child")
    assert parent.source_project == SourceProject.GOOGLE_TASKS_JSON
    assert parent.metadata["task_list"] == "Work"
    assert parent.metadata["notes"] == "See docs"
    assert parent.metadata["due_at"] == "2025-01-05T00:00:00+00:00"
    assert parent.metadata["links"] == [{"type": "email", "link": "https://example.com"}]
    assert child.metadata["parent_task_id"] == "parent"
    assert child.metadata["completed_at"] == "2025-01-03T00:00:00+00:00"
    assert child.updated_at == datetime(2025, 1, 3, tzinfo=timezone.utc)
    assert get_adapter("google_tasks_json", path=str(export)).name == "google_tasks_json"


def test_google_tasks_json_accepts_tasks_without_notes_and_filters(tmp_path):
    export = tmp_path / "tasks.json"
    export.write_text(json.dumps({"title": "Personal", "items": [{"id": "task-1", "title": "Buy milk"}]}), encoding="utf-8")

    result = GoogleTasksJsonAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Buy milk"]
    assert "notes" not in result.units[0].metadata
    assert GoogleTasksJsonAdapter(path=str(export)).ingest(entity_types=["list"]).units == []
