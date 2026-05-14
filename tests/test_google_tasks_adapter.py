from __future__ import annotations

import json
from pathlib import Path

from graph.adapters.google_tasks import GoogleTasksAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_task_list(title: str = "My Tasks", items: list | None = None) -> dict:
    return {
        "kind": "tasks#taskList",
        "title": title,
        "items": items or [],
    }


def test_parses_task_list_and_tasks(tmp_path):
    data = _make_task_list("Work", items=[
        {"id": "t1", "title": "Finish report", "status": "needsAction"},
        {"id": "t2", "title": "Send email", "status": "completed", "notes": "To Bob"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()

    lists = [u for u in result.units if u.source_entity_type == "task_list"]
    tasks = [u for u in result.units if u.source_entity_type == "task"]
    assert len(lists) == 1
    assert lists[0].title == "Work"
    assert len(tasks) == 2


def test_task_status_as_tag(tmp_path):
    data = _make_task_list(items=[
        {"id": "t1", "title": "Done task", "status": "completed"},
        {"id": "t2", "title": "Open task", "status": "needsAction"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()

    tasks = [u for u in result.units if u.source_entity_type == "task"]
    tags_by_title = {u.title: u.tags for u in tasks}
    assert "completed" in tags_by_title["Done task"]
    assert "needsaction" in tags_by_title["Open task"]


def test_subtask_hierarchy_creates_edges(tmp_path):
    data = _make_task_list(items=[
        {"id": "parent1", "title": "Parent task", "status": "needsAction"},
        {"id": "child1", "title": "Sub task", "status": "needsAction", "parent": "parent1"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()

    contains = [e for e in result.edges if e.relation == EdgeRelation.CONTAINS]
    subtask_edges = [e for e in contains if e.metadata["relation_type"] == "subtask"]
    assert len(subtask_edges) == 1


def test_task_notes_as_content(tmp_path):
    data = _make_task_list(items=[
        {"id": "t1", "title": "With notes", "notes": "Some detailed notes", "status": ""},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()
    task = [u for u in result.units if u.source_entity_type == "task"][0]
    assert task.content == "Some detailed notes"


def test_empty_task_list(tmp_path):
    data = _make_task_list(items=[])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()
    lists = [u for u in result.units if u.source_entity_type == "task_list"]
    tasks = [u for u in result.units if u.source_entity_type == "task"]
    assert len(lists) == 1
    assert len(tasks) == 0


def test_no_path_returns_empty():
    result = GoogleTasksAdapter().ingest()
    assert len(result.units) == 0


def test_entity_type_filtering(tmp_path):
    data = _make_task_list(items=[
        {"id": "t1", "title": "A task", "status": "needsAction"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest(entity_types=["task"])
    assert all(u.source_entity_type == "task" for u in result.units)


def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("google_tasks", path="/tmp/fake")
    assert adapter.name == "google_tasks"


def test_directory_with_multiple_lists(tmp_path):
    _write_json(tmp_path / "list1.json", _make_task_list("Work", items=[
        {"id": "t1", "title": "Task 1", "status": "needsAction"},
    ]))
    _write_json(tmp_path / "list2.json", _make_task_list("Personal", items=[
        {"id": "t2", "title": "Task 2", "status": "completed"},
    ]))

    result = GoogleTasksAdapter(path=str(tmp_path)).ingest()
    lists = [u for u in result.units if u.source_entity_type == "task_list"]
    tasks = [u for u in result.units if u.source_entity_type == "task"]
    assert len(lists) == 2
    assert len(tasks) == 2


def test_task_metadata_fields(tmp_path):
    data = _make_task_list(items=[
        {"id": "t1", "title": "Due task", "status": "needsAction",
         "due": "2024-06-15T00:00:00.000Z", "updated": "2024-06-10T12:00:00.000Z"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()
    task = [u for u in result.units if u.source_entity_type == "task"][0]
    assert task.metadata["due"] == "2024-06-15T00:00:00.000Z"
    assert task.metadata["list_title"] == "My Tasks"


def test_task_list_membership_edges_are_deterministic(tmp_path):
    data = _make_task_list("Work", items=[
        {"id": "t1", "title": "Task 1", "status": "needsAction"},
        {"id": "t2", "title": "Task 2", "status": "completed"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    first = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()
    second = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()

    task_list = next(u for u in first.units if u.source_entity_type == "task_list")
    membership_edges = [
        edge
        for edge in first.edges
        if edge.metadata["relation_type"] == "task_list_membership"
    ]
    assert len(membership_edges) == 2
    assert {edge.to_unit_id for edge in membership_edges} == {task_list.source_id}
    assert {edge.from_unit_id for edge in membership_edges} == {
        u.source_id for u in first.units if u.source_entity_type == "task"
    }
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert [edge.metadata for edge in first.edges] == [edge.metadata for edge in second.edges]


def test_missing_task_list_titles_use_stable_distinct_fallbacks(tmp_path):
    _write_json(tmp_path / "first.json", {"kind": "tasks#taskList", "items": []})
    _write_json(tmp_path / "second.json", {"kind": "tasks#taskList", "title": "", "items": []})

    result = GoogleTasksAdapter(path=str(tmp_path)).ingest()

    lists = [u for u in result.units if u.source_entity_type == "task_list"]
    assert sorted(u.title for u in lists) == ["first", "second"]
    assert len({u.source_id for u in lists}) == 2


def test_task_recurrence_metadata_is_normalized_and_omitted_when_empty(tmp_path):
    data = _make_task_list(items=[
        {
            "id": "t1",
            "title": "Weekly sync",
            "status": "needsAction",
            "recurrence": {"frequency": "weekly", "interval": 1, "until": ""},
        },
        {"id": "t2", "title": "Daily review", "status": "needsAction", "repeat": "RRULE:FREQ=DAILY"},
        {"id": "t3", "title": "One-off", "status": "needsAction", "recurrence": {}},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest()

    tasks = {u.title: u for u in result.units if u.source_entity_type == "task"}
    assert tasks["Weekly sync"].metadata["recurrence"] == {"frequency": "weekly", "interval": 1}
    assert tasks["Daily review"].metadata["recurrence"] == "RRULE:FREQ=DAILY"
    assert "recurrence" not in tasks["One-off"].metadata


def test_task_due_day_aggregates_due_tasks_and_edges(tmp_path):
    data = _make_task_list("Work", items=[
        {"id": "t1", "title": "First", "status": "needsAction", "due": "2024-06-15T00:00:00.000Z"},
        {"id": "t2", "title": "Second", "status": "completed", "due": "2024-06-15T12:30:00.000Z"},
        {"id": "t3", "title": "Undated", "status": "needsAction"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    result = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest(entity_types=["task", "task_due_day"])

    assert "task_due_day" in GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).entity_types
    due_day = next(unit for unit in result.units if unit.source_entity_type == "task_due_day")
    tasks = [unit for unit in result.units if unit.source_entity_type == "task"]
    due_tasks = [unit for unit in tasks if unit.metadata.get("due")]
    assert due_day.source_id == "google_tasks:task_due_day:2024-06-15"
    assert due_day.metadata == {
        "due_date": "2024-06-15",
        "task_count": 2,
        "completed_count": 1,
        "incomplete_count": 1,
        "list_titles": ["Work"],
        "task_source_ids": sorted(unit.source_id for unit in due_tasks),
    }
    due_day_edges = [
        edge for edge in result.edges if edge.metadata["relation_type"] == "task_due_day_contains_task"
    ]
    assert len(due_day_edges) == 2
    assert {edge.from_unit_id for edge in due_day_edges} == {due_day.source_id}
    assert {edge.to_unit_id for edge in due_day_edges} == {unit.source_id for unit in due_tasks}
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in due_day_edges)


def test_task_due_day_filtering_omits_edges_without_tasks(tmp_path):
    data = _make_task_list(items=[
        {"id": "t1", "title": "First", "status": "needsAction", "due": "2024-06-15"},
        {"id": "t2", "title": "Undated", "status": "needsAction"},
    ])
    _write_json(tmp_path / "tasks.json", data)

    due_days = GoogleTasksAdapter(path=str(tmp_path / "tasks.json")).ingest(entity_types=["task_due_day"])

    assert [unit.source_entity_type for unit in due_days.units] == ["task_due_day"]
    assert due_days.edges == []
