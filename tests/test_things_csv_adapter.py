from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.things_csv import ThingsCsvAdapter
from graph.types.enums import EdgeRelation, SourceProject


def test_things_csv_ingests_task_states_and_metadata(tmp_path):
    export = tmp_path / "things.csv"
    rows = [
        {
            "Title": "Open task",
            "Notes": "Open notes",
            "Area": "Personal",
            "Project": "Inbox",
            "Tags": "home,admin",
            "Creation Date": "2025-01-01",
            "Start Date": "",
            "Deadline": "",
            "Completion Date": "",
            "Canceled": "",
            "Checklist": "Subtask A\nSubtask B",
            "UUID": "open-1",
        },
        {
            "Title": "Completed task",
            "Notes": "",
            "Area": "Work",
            "Project": "Launch",
            "Tags": "work",
            "Creation Date": "2025-01-02",
            "Start Date": "",
            "Deadline": "",
            "Completion Date": "2025-01-03T10:00:00+00:00",
            "Canceled": "",
            "Checklist": "",
            "UUID": "done-1",
        },
        {
            "Title": "Scheduled task",
            "Notes": "",
            "Area": "",
            "Project": "",
            "Tags": "",
            "Creation Date": "2025-01-04",
            "Start Date": "2025-01-05",
            "Deadline": "2025-01-06",
            "Completion Date": "",
            "Canceled": "",
            "Checklist": "",
            "UUID": "scheduled-1",
        },
        {
            "Title": "Canceled task",
            "Notes": "",
            "Area": "",
            "Project": "",
            "Tags": "",
            "Creation Date": "2025-01-07",
            "Start Date": "",
            "Deadline": "",
            "Completion Date": "",
            "Canceled": "true",
            "Checklist": "",
            "UUID": "canceled-1",
        },
    ]
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = ThingsCsvAdapter(path=str(export)).ingest(entity_types=["task"])

    assert len(result.units) == 4
    by_title = {unit.title: unit for unit in result.units}
    open_task = by_title["Open task"]
    assert open_task.source_project == SourceProject.THINGS_CSV
    assert open_task.source_id.startswith("things_csv:")
    assert open_task.content == "Open task\n\nOpen notes"
    assert open_task.metadata["area"] == "Personal"
    assert open_task.metadata["project"] == "Inbox"
    assert open_task.metadata["tags"] == ["home", "admin"]
    assert open_task.metadata["status"] == "open"
    assert open_task.metadata["checklist"] == ["Subtask A", "Subtask B"]
    assert open_task.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert by_title["Completed task"].metadata["status"] == "completed"
    assert by_title["Scheduled task"].metadata["status"] == "scheduled"
    assert by_title["Scheduled task"].metadata["deadline"] == "2025-01-06T00:00:00+00:00"
    assert by_title["Canceled task"].metadata["status"] == "canceled"
    assert by_title["Canceled task"].metadata["canceled"] is True


def test_things_csv_adapter_is_registered():
    assert "things_csv" in list_adapters()
    assert get_adapter("things_csv", path="/tmp/things.csv").name == "things_csv"


def test_things_csv_emits_project_and_area_aggregates_and_edges(tmp_path):
    export = tmp_path / "things.csv"
    rows = [
        {"Title": "Open task", "Area": "Work", "Project": "Launch", "Creation Date": "2025-01-01", "UUID": "open-1"},
        {
            "Title": "Done task",
            "Area": "Work",
            "Project": "Launch",
            "Creation Date": "2025-01-02",
            "Completion Date": "2025-01-03",
            "UUID": "done-1",
        },
        {"Title": "Canceled task", "Area": "Home", "Project": "Move", "Creation Date": "2025-01-04", "Canceled": "true", "UUID": "cancel-1"},
    ]
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list({key: None for row in rows for key in row.keys()}.keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = ThingsCsvAdapter(path=str(export)).ingest(entity_types=["task", "project", "area"])

    assert ThingsCsvAdapter(path=str(export)).entity_types == ["task", "project", "area", "deadline_bucket"]
    launch = next(unit for unit in result.units if unit.source_entity_type == "project" and unit.title == "Launch")
    launch_tasks = [unit for unit in result.units if unit.source_entity_type == "task" and unit.metadata["project"] == "Launch"]
    assert launch.metadata["task_count"] == 2
    assert launch.metadata["open_count"] == 1
    assert launch.metadata["completed_count"] == 1
    assert launch.metadata["canceled_count"] == 0
    assert launch.metadata["task_source_ids"] == sorted(unit.source_id for unit in launch_tasks)
    assert launch.metadata["first_created_at"] == "2025-01-01T00:00:00+00:00"
    assert launch.metadata["latest_updated_at"] == "2025-01-03T00:00:00+00:00"
    assert {edge.to_unit_id for edge in result.edges if edge.from_unit_id == launch.source_id} == {unit.source_id for unit in launch_tasks}

    area_only = ThingsCsvAdapter(path=str(export)).ingest(entity_types=["area"])
    assert {unit.source_entity_type for unit in area_only.units} == {"area"}
    assert area_only.edges == []


def test_things_csv_emits_deadline_bucket_aggregates_and_edges(tmp_path):
    export = tmp_path / "things.csv"
    rows = [
        {
            "Title": "Overdue task",
            "Creation Date": "2025-01-01",
            "Deadline": "2025-01-09",
            "UUID": "overdue-1",
        },
        {
            "Title": "Today task",
            "Creation Date": "2025-01-02",
            "Deadline": "2025-01-10",
            "UUID": "today-1",
        },
        {
            "Title": "Upcoming task",
            "Creation Date": "2025-01-03",
            "Deadline": "2025-01-15",
            "UUID": "upcoming-1",
        },
        {
            "Title": "Later task",
            "Creation Date": "2025-01-04",
            "Deadline": "2025-02-01",
            "UUID": "later-1",
        },
        {
            "Title": "No deadline task",
            "Creation Date": "2025-01-05",
            "Deadline": "",
            "UUID": "none-1",
        },
        {
            "Title": "Completed today task",
            "Creation Date": "2025-01-06",
            "Deadline": "2025-01-10",
            "Completion Date": "2025-01-10",
            "UUID": "done-today-1",
        },
    ]
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list({key: None for row in rows for key in row.keys()}.keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = ThingsCsvAdapter(
        path=str(export),
        now=datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc),
    ).ingest(entity_types=["task", "deadline_bucket"])

    buckets = [unit for unit in result.units if unit.source_entity_type == "deadline_bucket"]
    tasks = [unit for unit in result.units if unit.source_entity_type == "task"]
    assert [unit.metadata["bucket"] for unit in buckets] == [
        "overdue",
        "today",
        "upcoming",
        "later",
        "no_deadline",
    ]

    today = next(unit for unit in buckets if unit.metadata["bucket"] == "today")
    today_tasks = [unit for unit in tasks if unit.metadata["deadline"] == "2025-01-10T00:00:00+00:00"]
    assert today.source_id == "things_csv:deadline_bucket:today"
    assert today.metadata["task_count"] == 2
    assert today.metadata["open_count"] == 1
    assert today.metadata["completed_count"] == 1
    assert today.metadata["first_deadline"] == "2025-01-10T00:00:00+00:00"
    assert today.metadata["latest_deadline"] == "2025-01-10T00:00:00+00:00"
    assert today.metadata["task_source_ids"] == sorted(unit.source_id for unit in today_tasks)

    no_deadline = next(unit for unit in buckets if unit.metadata["bucket"] == "no_deadline")
    assert no_deadline.metadata["first_deadline"] is None
    assert no_deadline.metadata["latest_deadline"] is None
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "deadline_bucket_contains_task"]) == len(tasks)
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)


def test_things_csv_deadline_bucket_filtering(tmp_path):
    export = tmp_path / "things.csv"
    export.write_text(
        "Title,Creation Date,Deadline,UUID\nTask,2025-01-01,2025-01-10,task-1\n",
        encoding="utf-8",
    )

    bucket_only = ThingsCsvAdapter(
        path=str(export),
        now=datetime(2025, 1, 10, tzinfo=timezone.utc),
    ).ingest(entity_types=["deadline_bucket"])
    task_only = ThingsCsvAdapter(
        path=str(export),
        now=datetime(2025, 1, 10, tzinfo=timezone.utc),
    ).ingest(entity_types=["task"])

    assert [unit.source_entity_type for unit in bucket_only.units] == ["deadline_bucket"]
    assert bucket_only.edges == []
    assert [unit.source_entity_type for unit in task_only.units] == ["task"]
    assert task_only.edges == []
