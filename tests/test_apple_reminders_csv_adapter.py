from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.apple_reminders_csv import AppleRemindersCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import EdgeRelation, SourceProject
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

    assert len([unit for unit in result.units if unit.source_entity_type == "reminder"]) == 2
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

    grocery = next(unit for unit in result.units if unit.source_entity_type == "list" and unit.title == "Groceries")
    assert grocery.metadata["reminder_count"] == 1
    assert grocery.metadata["open_count"] == 1
    assert grocery.metadata["completed_count"] == 0
    assert grocery.metadata["earliest_due_date"] == "2025-01-03T09:00:00+00:00"
    assert grocery.metadata["latest_due_date"] == "2025-01-03T09:00:00+00:00"
    assert grocery.metadata["source_files"] == ["reminders.csv"]
    edge = next(edge for edge in result.edges if edge.to_unit_id == open_unit.source_id)
    assert edge.from_unit_id == grocery.source_id
    assert edge.relation == EdgeRelation.CONTAINS


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

    result = AppleRemindersCsvAdapter(path=str(export)).ingest(since=since, entity_types=["reminder"])
    titles = {unit.title for unit in result.units}

    assert titles == {"New due", "No optional fields"}
    missing = next(unit for unit in result.units if unit.title == "No optional fields")
    assert missing.metadata["priority"] == ""
    assert missing.metadata["url"] == ""
    assert missing.tags == ["reminder", "open"]
    assert AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["task"]).units == []


def test_apple_reminders_csv_entity_filters_for_lists_and_reminders(tmp_path):
    export = tmp_path / "reminders.csv"
    _write_csv(
        export,
        [
            {"Title": "Open", "List Name": "Work", "Completed": "No", "Created Date": "2025-01-01", "Due Date": "2025-01-02"},
            {"Title": "Done", "List Name": "Work", "Completed": "Yes", "Created Date": "2025-01-03"},
            {"Title": "Inbox item", "List Name": "", "Completed": "No", "Created Date": "2025-01-04"},
        ],
    )

    list_only = AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["list"])
    reminder_only = AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["reminder"])
    combined = AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["list", "reminder"])

    assert [unit.source_entity_type for unit in list_only.units] == ["list"]
    assert list_only.units[0].metadata["open_count"] == 1
    assert list_only.units[0].metadata["completed_count"] == 1
    assert list_only.edges == []
    assert {unit.source_entity_type for unit in reminder_only.units} == {"reminder"}
    assert reminder_only.edges == []
    assert len(combined.edges) == 2


def test_apple_reminders_csv_list_metadata_uses_due_date_bounds_and_aliases(tmp_path):
    export = tmp_path / "reminders.csv"
    _write_csv(
        export,
        [
            {"Title": "Later", "Reminder List": "Work", "Completed": "No", "Due Date": "2025-02-03"},
            {"Title": "Earlier", "Reminder List": "Work", "Completed": "Yes", "Due Date": "2025-01-03"},
        ],
    )

    result = AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["list"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["list_name"] == "Work"
    assert unit.metadata["reminder_count"] == 2
    assert unit.metadata["completed_count"] == 1
    assert unit.metadata["open_count"] == 1
    assert unit.metadata["earliest_due_date"] == "2025-01-03T00:00:00+00:00"
    assert unit.metadata["latest_due_date"] == "2025-02-03T00:00:00+00:00"
    assert unit.metadata["first_due_date"] == "2025-01-03T00:00:00+00:00"
    assert unit.metadata["last_updated_date"] == "2025-02-03T00:00:00+00:00"


def test_apple_reminders_csv_list_summaries_dedupe_normalized_names(tmp_path):
    export = tmp_path / "reminders.csv"
    _write_csv(
        export,
        [
            {"Title": "One", "List": "Work", "list_name": "", "Completed": "No", "Due Date": "2025-01-02", "Completion Date": ""},
            {"Title": "Two", "List": "", "list_name": " work ", "Completed": "Yes", "Due Date": "2025-01-03", "Completion Date": "2025-01-04"},
            {"Title": "No list", "List": "", "list_name": "", "Completed": "No", "Due Date": "2025-01-05", "Completion Date": ""},
        ],
    )

    result = AppleRemindersCsvAdapter(path=str(export)).ingest(entity_types=["list", "reminder"])

    lists = [unit for unit in result.units if unit.source_entity_type == "list"]
    assert len(lists) == 1
    assert lists[0].metadata["list_name"] == "Work"
    assert lists[0].metadata["reminder_count"] == 2
    assert lists[0].metadata["completed_count"] == 1
    assert lists[0].metadata["open_count"] == 1
    assert len(result.edges) == 2


def test_apple_reminders_csv_directory_and_registry(tmp_path):
    _write_csv(tmp_path / "one.csv", [{"Title": "One", "Created Date": "2025-01-01"}])
    _write_csv(tmp_path / "two.csv", [{"Title": "Two", "Created Date": "2025-01-02"}])

    result = AppleRemindersCsvAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["One", "Two"]
    assert "apple_reminders_csv" in list_adapters()
    assert get_adapter("apple_reminders_csv", path=str(tmp_path)).name == "apple_reminders_csv"
