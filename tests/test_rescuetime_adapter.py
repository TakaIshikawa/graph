from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.rescuetime import RescueTimeAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_rescuetime_csv_ingests_activities_with_full_metadata(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Time Spent (seconds),Activity,Category,Productivity",
                "2025-01-15,3600,Visual Studio Code,Software Development,2",
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.RESCUETIME
    assert unit.source_id.startswith("rescuetime:")
    assert unit.source_entity_type == "activity"
    assert unit.title == "Visual Studio Code on 2025-01-15"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Activity: Visual Studio Code" in unit.content
    assert "Category: Software Development" in unit.content
    assert "Time Spent: 1h" in unit.content
    assert "Productivity: 2" in unit.content
    assert unit.metadata == {
        "activity": "Visual Studio Code",
        "category": "Software Development",
        "time_spent_seconds": "3600",
        "productivity": "2",
        "date": "2025-01-15",
    }
    assert unit.tags == ["software-development", "productivity-2"]
    assert unit.created_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_rescuetime_csv_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity",
                "2025-01-20,Chrome",
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Chrome on 2025-01-20"
    assert unit.metadata["activity"] == "Chrome"
    assert unit.metadata["category"] == ""
    assert unit.metadata["time_spent_seconds"] == ""
    assert unit.metadata["productivity"] == ""
    assert unit.tags == []


def test_rescuetime_formats_duration_correctly(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity,Time Spent (seconds)",
                "2025-01-01,App1,45",  # 45 seconds
                "2025-01-02,App2,300",  # 5 minutes
                "2025-01-03,App3,7200",  # 2 hours
                "2025-01-04,App4,5430",  # 1h 30m
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 4
    assert "Time Spent: 45s" in result.units[0].content
    assert "Time Spent: 5m" in result.units[1].content
    assert "Time Spent: 2h" in result.units[2].content
    assert "Time Spent: 1h 30m" in result.units[3].content


def test_rescuetime_json_ingests_activities_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "rescuetime.json"
    export.write_text(
        json.dumps(
            {
                "activities": [
                    {
                        "date": "2025-01-25T10:00:00Z",
                        "activity": "Slack",
                        "category": "Communication & Scheduling",
                        "time_spent": "1800",
                        "productivity": "0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = RescueTimeAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Slack on 2025-01-25T10:00:00Z"
    assert unit.metadata["activity"] == "Slack"
    assert unit.metadata["category"] == "Communication & Scheduling"
    assert unit.metadata["time_spent_seconds"] == "1800"
    assert unit.metadata["productivity"] == "0"
    assert unit.tags == ["communication-&-scheduling", "productivity-0"]
    assert unit.created_at == datetime(2025, 1, 25, 10, 0, tzinfo=timezone.utc)


def test_rescuetime_since_filters_activities_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity",
                "2025-01-01,Old Activity",
                "2025-01-02,Equal Activity",
                "2025-01-03,New Activity",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="rescuetime",
        source_entity_type="activity",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = RescueTimeAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert "New Activity" in result.units[0].title


def test_rescuetime_respects_entity_types(tmp_path):
    export = tmp_path / "rescuetime.json"
    export.write_text(
        json.dumps([{"activity": "Test App", "date": "2025-01-01"}]), encoding="utf-8"
    )

    result = RescueTimeAdapter(path=str(export)).ingest(entity_types=["book"])

    assert result.units == []
    assert result.edges == []


def test_rescuetime_skips_activities_without_name(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity",
                "2025-01-01,",  # Empty activity
                "2025-01-02,Valid Activity",
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["activity"] == "Valid Activity"


def test_rescuetime_handles_malformed_csv(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text("This is not a valid CSV file\n<<>><>", encoding="utf-8")

    result = RescueTimeAdapter(path=str(export)).ingest()

    # Should return empty result without crashing
    assert result.units == []


def test_rescuetime_handles_empty_export(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity",
                # No data rows
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert result.units == []


def test_rescuetime_handles_nonexistent_file():
    result = RescueTimeAdapter(path="/nonexistent/path.csv").ingest()

    assert result.units == []


def test_rescuetime_creates_unique_source_ids_for_same_activity(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity,Time Spent (seconds)",
                "2025-01-01,Chrome,1800",
                "2025-01-02,Chrome,3600",
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    # Source IDs should be different because they include date and time spent
    assert result.units[0].source_id != result.units[1].source_id


def test_rescuetime_parses_productivity_scores(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Activity,Productivity",
                "2025-01-01,App1,Very Productive",
                "2025-01-02,App2,Productive",
                "2025-01-03,App3,Neutral",
                "2025-01-04,App4,Distracting",
            ]
        ),
        encoding="utf-8",
    )

    result = RescueTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 4
    assert result.units[0].tags == ["productivity-very-productive"]
    assert result.units[1].tags == ["productivity-productive"]
    assert result.units[2].tags == ["productivity-neutral"]
    assert result.units[3].tags == ["productivity-distracting"]


def test_rescuetime_adapter_is_registered():
    assert "rescuetime" in list_adapters()
    adapter = get_adapter("rescuetime", path="/tmp/rescuetime.csv")
    assert adapter.name == "rescuetime"
