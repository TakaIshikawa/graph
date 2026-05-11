from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.apple_reminders_csv import AppleRemindersCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_apple_reminders_csv_ingests_open_and_completed_reminders(tmp_path):
    export = tmp_path / "reminders.csv"
    _write_csv(
        export,
        [
            {
                "Title": "Buy milk",
                "Notes": "Whole milk",
                "List Name": "Groceries",
                "Completed": "No",
                "Priority": "High",
                "Due Date": "2025-01-03T09:00:00Z",
                "Completion Date": "",
                "Created Date": "2025-01-01",
                "URL": "https://example.com/milk",
                "ID": "open-1",
            },
            {
                "Title": "File receipt",
                "Notes": "",
                "List Name": "Admin",
                "Completed": "Yes",
                "Priority": "Low",
                "Due Date": "01/04/2025",
                "Completion Date": "Jan 5, 2025 at 3:30 PM",
                "Created Date": "01/02/2025 08:15",
                "URL": "",
                "ID": "done-1",
            },
        ],
    )

    result = AppleRemindersCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    by_title = {unit.title: unit for unit in result.units}
    open_unit = by_title["Buy milk"]
    assert open_unit.source_project == SourceProject.APPLE_REMINDERS_CSV
    assert open_unit.source_entity_type == "reminder"
    assert open_unit.metadata["notes"] == "Whole milk"
    assert open_unit.metadata["list_name"] == "Groceries"
    assert open_unit.metadata["status"] == "open"
    assert open_unit.metadata["priority"] == "High"
    assert open_unit.metadata["due_date"] == "2025-01-03T09:00:00+00:00"
    assert open_unit.metadata["url"] == "https://example.com/milk"
    assert open_unit.tags == ["reminder", "open", "Groceries"]
    done = by_title["File receipt"]
    assert done.metadata["status"] == "completed"
    assert done.metadata["completion_date"] == "2025-01-05T15:30:00+00:00"
    assert "completed" in done.tags


def test_apple_reminders_csv_filters_and_missing_optional_fields(tmp_path):
    export = tmp_path / "reminders.csv"
    _write_csv(
        export,
        [
            {"Title": "Old", "Created Date": "2025-01-01", "Due Date": "", "Completion Date": "", "Completed": ""},
            {"Title": "New due", "Created Date": "", "Due Date": "2025-02-01", "Completion Date": "", "Completed": ""},
            {"Title": "No optional fields", "Created Date": "2025-03-01", "Due Date": "", "Completion Date": "", "Completed": ""},
        ],
    )
    since = SyncState(
        source_project="apple_reminders_csv",
        source_entity_type="reminder",
        last_sync_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )

    result = AppleRemindersCsvAdapter(path=str(export)).ingest(since=since)
    titles = {unit.title for unit in result.units}

    assert titles == {"New due", "No optional fields"}
    missing = next(unit for unit in result.units if unit.title == "No optional fields")
    assert missing.metadata["priority"] == ""
    assert missing.metadata["url"] == ""
    assert missing.tags == ["reminder", "open"]
    assert AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["task"]).units == []


def test_apple_reminders_csv_directory_and_registry(tmp_path):
    _write_csv(tmp_path / "one.csv", [{"Title": "One", "Created Date": "2025-01-01"}])
    _write_csv(tmp_path / "two.csv", [{"Title": "Two", "Created Date": "2025-01-02"}])

    result = AppleRemindersCsvAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["One", "Two"]
    assert "apple_reminders_csv" in list_adapters()
    assert get_adapter("apple_reminders_csv", path=str(tmp_path)).name == "apple_reminders_csv"
