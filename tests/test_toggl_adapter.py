from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.toggl import TogglAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_toggl_csv_ingests_time_entries_with_full_metadata(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Project,Description,Start date,Start time,End date,End time,Duration,Tags,Billable",
                "Project A,Feature implementation,2025-01-15,09:00:00,2025-01-15,11:30:00,02:30:00,\"development,backend\",Yes",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.TOGGL
    assert unit.source_id.startswith("toggl:")
    assert unit.source_entity_type == "time_entry"
    assert unit.title == "Feature implementation (Project A)"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Task: Feature implementation" in unit.content
    assert "Project: Project A" in unit.content
    assert "Duration: 02:30:00" in unit.content
    assert "Tags: development, backend" in unit.content
    assert "Billable: Yes" in unit.content
    assert unit.metadata == {
        "description": "Feature implementation",
        "project": "Project A",
        "start": "2025-01-15",
        "end": "2025-01-15",
        "duration": "02:30:00",
        "tags": ["development", "backend"],
        "billable": "Yes",
    }
    assert unit.tags == ["development", "backend"]
    assert unit.created_at == datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 15, 11, 30, tzinfo=timezone.utc)


def test_toggl_csv_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Start date,Start time",
                "Simple task,2025-01-20,14:00:00",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Simple task"
    assert unit.metadata["description"] == "Simple task"
    assert unit.metadata["project"] == ""
    assert unit.metadata["duration"] == ""
    assert unit.metadata["tags"] == []
    assert unit.metadata["billable"] == ""
    assert unit.tags == []


def test_toggl_handles_project_only_entries(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Project,Start date,Start time",
                "Project X,2025-01-10,10:00:00",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Time entry in Project X"
    assert unit.metadata["project"] == "Project X"


def test_toggl_json_ingests_time_entries_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "toggl.json"
    export.write_text(
        json.dumps(
            {
                "time_entries": [
                    {
                        "description": "Code review",
                        "project": "Project B",
                        "start": "2025-01-25T10:00:00Z",
                        "end": "2025-01-25T11:00:00Z",
                        "duration": "01:00:00",
                        "tags": "review,quality",
                        "billable": "No",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = TogglAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Code review (Project B)"
    assert unit.metadata["description"] == "Code review"
    assert unit.metadata["project"] == "Project B"
    assert unit.metadata["duration"] == "01:00:00"
    assert unit.metadata["tags"] == ["review", "quality"]
    assert unit.metadata["billable"] == "No"
    assert unit.tags == ["review", "quality"]
    assert "Billable: No" in unit.content


def test_toggl_since_filters_entries_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Start date,Start time",
                "Old task,2025-01-01,09:00:00",
                "Equal task,2025-01-02,09:00:00",
                "New task,2025-01-03,09:00:00",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="toggl",
        source_entity_type="time_entry",
        last_sync_at=datetime(2025, 1, 2, 9, 0, tzinfo=timezone.utc),
    )

    result = TogglAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert "New task" in result.units[0].title


def test_toggl_respects_entity_types(tmp_path):
    export = tmp_path / "toggl.json"
    export.write_text(
        json.dumps([{"description": "Test task", "project": "Test"}]), encoding="utf-8"
    )

    result = TogglAdapter(path=str(export)).ingest(entity_types=["activity"])

    assert result.units == []
    assert result.edges == []


def test_toggl_skips_entries_without_description_and_project(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Project,Start date",
                ",,2025-01-01",  # Empty description and project
                "Valid task,,2025-01-02",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["description"] == "Valid task"


def test_toggl_handles_malformed_csv(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text("This is not a valid CSV file\n<<>><>", encoding="utf-8")

    result = TogglAdapter(path=str(export)).ingest()

    # Should return empty result without crashing
    assert result.units == []


def test_toggl_handles_empty_export(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Project",
                # No data rows
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert result.units == []


def test_toggl_handles_nonexistent_file():
    result = TogglAdapter(path="/nonexistent/path.csv").ingest()

    assert result.units == []


def test_toggl_parses_tags_correctly(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Tags",
                "Task 1,\"development,testing,documentation\"",
                "Task 2,meeting",
                "Task 3,",  # Empty tags
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    assert result.units[0].tags == ["development", "testing", "documentation"]
    assert result.units[1].tags == ["meeting"]
    assert result.units[2].tags == []


def test_toggl_handles_billable_status(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Billable",
                "Task 1,Yes",
                "Task 2,No",
                "Task 3,true",
                "Task 4,1",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 4
    assert "Billable: Yes" in result.units[0].content
    assert "Billable: No" in result.units[1].content
    assert "Billable: Yes" in result.units[2].content
    assert "Billable: Yes" in result.units[3].content


def test_toggl_creates_unique_source_ids_for_similar_entries(tmp_path):
    export = tmp_path / "toggl.csv"
    export.write_text(
        "\n".join(
            [
                "Description,Project,Start date,Start time,Duration",
                "Same task,Same project,2025-01-01,09:00:00,01:00:00",
                "Same task,Same project,2025-01-02,09:00:00,01:00:00",
            ]
        ),
        encoding="utf-8",
    )

    result = TogglAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    # Source IDs should be different because they include start time
    assert result.units[0].source_id != result.units[1].source_id


def test_toggl_adapter_is_registered():
    assert "toggl" in list_adapters()
    adapter = get_adapter("toggl", path="/tmp/toggl.csv")
    assert adapter.name == "toggl"
