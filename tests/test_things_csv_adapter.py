from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.things_csv import ThingsCsvAdapter
from graph.types.enums import SourceProject


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

    result = ThingsCsvAdapter(path=str(export)).ingest()

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
