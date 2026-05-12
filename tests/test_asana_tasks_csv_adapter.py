from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.asana_tasks_csv import AsanaTasksCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_asana_tasks_csv_ingests_tasks_metadata_tags_and_parent_edges(tmp_path):
    export = tmp_path / "asana.csv"
    _write_csv(
        export,
        [
            {
                "Task ID": "1",
                "Name": "Plan import",
                "Notes": "Parent task",
                "Assignee": "Ada",
                "Projects": "Imports, Graph",
                "Tags": "backend;csv",
                "Created At": "2026-05-01T10:00:00Z",
                "Modified At": "2026-05-02T10:00:00Z",
                "Due Date": "2026-05-05",
                "Completed At": "",
                "Parent Task ID": "",
                "Task URL": "https://app.asana.com/0/1/1",
            },
            {
                "Task ID": "2",
                "Name": "Write adapter",
                "Notes": "Child task",
                "Assignee": "Grace",
                "Projects": "Imports",
                "Tags": "csv",
                "Created At": "2026-05-01T11:00:00Z",
                "Modified At": "2026-05-03T10:00:00Z",
                "Due Date": "",
                "Completed At": "2026-05-04T10:00:00Z",
                "Parent Task ID": "1",
                "Task URL": "",
            },
        ],
    )

    result = AsanaTasksCsvAdapter(path=str(export)).ingest()

    tasks = [unit for unit in result.units if unit.source_entity_type == "task"]
    assert [unit.source_id for unit in tasks] == ["asana_tasks_csv:1", "asana_tasks_csv:2"]
    parent = tasks[0]
    child = tasks[1]
    assert parent.source_project == SourceProject.ASANA_TASKS_CSV
    assert parent.source_entity_type == "task"
    assert parent.metadata["assignee"] == "Ada"
    assert parent.metadata["projects"] == ["Imports", "Graph"]
    assert parent.metadata["tags"] == ["backend", "csv"]
    assert parent.metadata["due_date"] == "2026-05-05T00:00:00+00:00"
    assert parent.updated_at == datetime(2026, 5, 2, 10, tzinfo=timezone.utc)
    assert child.metadata["status"] == "completed"
    assert {"asana", "task", "Imports", "csv"}.issubset(set(child.tags))
    parent_edge = next(edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS)
    assert parent_edge.from_unit_id == parent.source_id
    assert parent_edge.to_unit_id == child.source_id


def test_asana_tasks_csv_since_and_entity_filters(tmp_path):
    export = tmp_path / "asana.csv"
    _write_csv(
        export,
        [
            {"Task ID": "old", "Name": "Old", "Created At": "2026-04-01", "Modified At": "2026-04-02"},
            {"Task ID": "new", "Name": "New", "Created At": "2026-04-01", "Modified At": "2026-05-02"},
        ],
    )
    since = SyncState(source_project="asana_tasks_csv", source_entity_type="task", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = AsanaTasksCsvAdapter(path=str(export)).ingest(since=since, entity_types=["task"])

    assert [unit.title for unit in result.units if unit.source_entity_type == "task"] == ["New"]
    assert AsanaTasksCsvAdapter(path=str(export)).ingest(entity_types=["unknown"]).units == []


def test_asana_tasks_csv_empty_files_and_registry_lookup(tmp_path):
    export = tmp_path / "empty.csv"
    _write_csv(export, [])

    assert AsanaTasksCsvAdapter(path=str(export)).ingest().units == []
    assert get_adapter("asana_tasks_csv", path=str(export)).name == "asana_tasks_csv"


def test_asana_tasks_csv_ingests_assignee_project_and_workspace_entities(tmp_path):
    export = tmp_path / "asana.csv"
    _write_csv(
        export,
        [
            {"Task ID": "1", "Name": "Assigned", "Assignee": "Ada", "Workspace": "Acme", "Projects": "Imports"},
            {"Task ID": "2", "Name": "Unassigned", "Assignee": "", "Workspace": "Research", "Projects": ""},
        ],
    )

    result = AsanaTasksCsvAdapter(path=str(export)).ingest()

    assert sorted(unit.title for unit in result.units if unit.source_entity_type == "assignee") == ["Ada"]
    assert sorted(unit.title for unit in result.units if unit.source_entity_type == "project") == ["Imports"]
    assert sorted(unit.title for unit in result.units if unit.source_entity_type == "workspace") == ["Acme", "Research"]
    assert {edge.metadata["relation_type"] for edge in result.edges} == {"task_assignee", "task_project", "task_workspace"}
