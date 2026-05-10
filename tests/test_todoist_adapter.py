from __future__ import annotations

from pathlib import Path

from graph.adapters.todoist import TodoistAdapter
from graph.types.enums import ContentType, SourceProject


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> Path:
    """Write a CSV file with the given rows."""
    import csv
    import io

    if columns is None:
        columns = ["TYPE", "CONTENT", "PRIORITY", "INDENT", "AUTHOR", "RESPONSIBLE", "DATE", "DATE_LANG", "TIMEZONE"]

    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Single task
# ---------------------------------------------------------------------------

def test_parse_single_task(tmp_path):
    csv_path = _write_csv(tmp_path / "tasks.csv", [
        {"TYPE": "task", "CONTENT": "Buy groceries", "PRIORITY": "1", "INDENT": "1",
         "AUTHOR": "alice", "RESPONSIBLE": "", "DATE": "2024-03-15", "DATE_LANG": "en", "TIMEZONE": "UTC"},
    ])

    result = TodoistAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Buy groceries"
    assert unit.source_project == SourceProject.TODOIST
    assert unit.source_entity_type == "task"
    assert unit.content_type == ContentType.ARTIFACT
    assert "p1" in unit.tags
    assert unit.metadata["priority"] == 1
    assert unit.metadata["due_date"] == "2024-03-15"
    assert unit.metadata["author"] == "alice"


# ---------------------------------------------------------------------------
# Multiple tasks with different priorities
# ---------------------------------------------------------------------------

def test_multiple_tasks_different_priorities(tmp_path):
    csv_path = _write_csv(tmp_path / "tasks.csv", [
        {"TYPE": "task", "CONTENT": "Urgent task", "PRIORITY": "1", "INDENT": "1"},
        {"TYPE": "task", "CONTENT": "Normal task", "PRIORITY": "3", "INDENT": "1"},
        {"TYPE": "task", "CONTENT": "Low task", "PRIORITY": "4", "INDENT": "1"},
    ])

    result = TodoistAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 3
    tags_by_title = {u.title: u.tags for u in result.units}
    assert "p1" in tags_by_title["Urgent task"]
    assert "p3" in tags_by_title["Normal task"]
    assert "p4" in tags_by_title["Low task"]


# ---------------------------------------------------------------------------
# Project-type rows
# ---------------------------------------------------------------------------

def test_project_type_rows(tmp_path):
    csv_path = _write_csv(tmp_path / "tasks.csv", [
        {"TYPE": "project", "CONTENT": "Work", "PRIORITY": "", "INDENT": "1"},
        {"TYPE": "task", "CONTENT": "Do something", "PRIORITY": "2", "INDENT": "2"},
    ])

    result = TodoistAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 2
    project_units = [u for u in result.units if u.source_entity_type == "project"]
    task_units = [u for u in result.units if u.source_entity_type == "task"]
    assert len(project_units) == 1
    assert project_units[0].title == "Work"
    assert len(task_units) == 1
    assert task_units[0].title == "Do something"


# ---------------------------------------------------------------------------
# Empty CSV
# ---------------------------------------------------------------------------

def test_empty_csv(tmp_path):
    csv_path = _write_csv(tmp_path / "empty.csv", [])

    result = TodoistAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


# ---------------------------------------------------------------------------
# Missing optional columns
# ---------------------------------------------------------------------------

def test_missing_optional_columns(tmp_path):
    csv_path = _write_csv(
        tmp_path / "minimal.csv",
        [{"TYPE": "task", "CONTENT": "A task"}],
        columns=["TYPE", "CONTENT"],
    )

    result = TodoistAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "A task"
    assert unit.tags == []
    assert "due_date" not in unit.metadata


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("todoist", path="/tmp/fake.csv")
    assert adapter.name == "todoist"


# ---------------------------------------------------------------------------
# Deterministic source IDs
# ---------------------------------------------------------------------------

def test_source_ids_deterministic(tmp_path):
    csv_path = _write_csv(tmp_path / "tasks.csv", [
        {"TYPE": "task", "CONTENT": "Same task", "PRIORITY": "1", "INDENT": "1", "DATE": "2024-01-01"},
    ])

    result1 = TodoistAdapter(path=str(csv_path)).ingest()
    result2 = TodoistAdapter(path=str(csv_path)).ingest()

    assert result1.units[0].source_id == result2.units[0].source_id


# ---------------------------------------------------------------------------
# No path returns empty
# ---------------------------------------------------------------------------

def test_no_path_returns_empty():
    result = TodoistAdapter().ingest()
    assert len(result.units) == 0


# ---------------------------------------------------------------------------
# Nonexistent path returns empty
# ---------------------------------------------------------------------------

def test_nonexistent_path_returns_empty():
    result = TodoistAdapter(path="/nonexistent/path.csv").ingest()
    assert len(result.units) == 0


# ---------------------------------------------------------------------------
# Entity type filtering
# ---------------------------------------------------------------------------

def test_entity_type_filtering(tmp_path):
    csv_path = _write_csv(tmp_path / "tasks.csv", [
        {"TYPE": "project", "CONTENT": "My Project", "PRIORITY": "", "INDENT": "1"},
        {"TYPE": "task", "CONTENT": "A task", "PRIORITY": "2", "INDENT": "2"},
    ])

    result = TodoistAdapter(path=str(csv_path)).ingest(entity_types=["task"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "task"
