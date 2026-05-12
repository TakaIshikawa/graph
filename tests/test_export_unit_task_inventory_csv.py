from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_task_inventory_csv import export_unit_task_inventory_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project A",
        source_id=f"source-{unit_id}",
        source_entity_type="task",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_task_inventory_csv_empty_input_has_header_only():
    assert export_unit_task_inventory_csv([]) == (
        "unit_id,title,source_project,source_entity_type,status,priority,due_date,completed_date,"
        "assignee_count,checklist_item_count,checklist_completed_count,is_complete\n"
    )


def test_unit_task_inventory_csv_units_without_task_metadata_have_header_only():
    assert export_unit_task_inventory_csv([unit("a", metadata={"topic": "planning"})]) == (
        "unit_id,title,source_project,source_entity_type,status,priority,due_date,completed_date,"
        "assignee_count,checklist_item_count,checklist_completed_count,is_complete\n"
    )


def test_unit_task_inventory_csv_normalizes_core_task_fields():
    text = export_unit_task_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "state": "In Progress",
                    "priority": " HIGH ",
                    "due": "2024-02-01T09:00:00Z",
                    "completed_at": "2024-02-02",
                    "assignees": ["Bob", " alice ", "Bob"],
                    "checklist_items": [
                        {"title": "One", "completed": True},
                        {"title": "Two", "status": "done"},
                        {"title": "Three", "completed": False},
                    ],
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "Project A",
            "source_entity_type": "task",
            "status": "in progress",
            "priority": "high",
            "due_date": "2024-02-01",
            "completed_date": "2024-02-02",
            "assignee_count": "2",
            "checklist_item_count": "3",
            "checklist_completed_count": "2",
            "is_complete": "true",
        }
    ]


def test_unit_task_inventory_csv_prefers_explicit_status_due_and_checklist_count():
    text = export_unit_task_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "status": "Done",
                    "state": "Open",
                    "due_date": "tomorrow",
                    "due": "2024-03-01",
                    "completed": "yes",
                    "assignee": "Casey",
                    "checklist_items": [{"completed": False}],
                    "checklist_completed_count": "7",
                },
            )
        ]
    )

    assert rows(text)[0] == {
        "unit_id": "a",
        "title": "Title a",
        "source_project": "Project A",
        "source_entity_type": "task",
        "status": "done",
        "priority": "",
        "due_date": "tomorrow",
        "assignee_count": "1",
        "completed_date": "",
        "checklist_item_count": "1",
        "checklist_completed_count": "7",
        "is_complete": "true",
    }


def test_unit_task_inventory_csv_sorts_by_due_date_source_and_unit_id_with_undated_last():
    units = [
        unit("b", metadata={"status": "open"}),
        unit("c", metadata={"due_date": "2024-02-01"}),
        unit("a", metadata={"due_date": "2024-01-01"}),
    ]

    assert export_unit_task_inventory_csv(units) == export_unit_task_inventory_csv(reversed(units))
    assert [row["unit_id"] for row in rows(export_unit_task_inventory_csv(units))] == ["a", "c", "b"]


def test_unit_task_inventory_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-task-inventory.csv"
    units = [unit("a", metadata={"status": "open"})]

    expected = export_unit_task_inventory_csv(units)
    stats = export_unit_task_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "task_unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
