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
    assert len(contains) == 1
    assert contains[0].metadata["relation_type"] == "subtask"


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
