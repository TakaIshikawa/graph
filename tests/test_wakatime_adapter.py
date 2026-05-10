from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.wakatime import WakaTimeAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_wakatime_json_ingests_coding_sessions_with_full_metadata(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "language": "Python",
                        "project": "my-app",
                        "file": "src/main.py",
                        "branch": "main",
                        "editor": "VS Code",
                        "duration": "3600",
                        "time": "2025-01-15T10:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.WAKATIME
    assert unit.source_id.startswith("wakatime:")
    assert unit.source_entity_type == "coding_session"
    assert unit.title == "Python in my-app on 2025-01-15T10:00:00Z"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Project: my-app" in unit.content
    assert "Language: Python" in unit.content
    assert "File: src/main.py" in unit.content
    assert "Editor: VS Code" in unit.content
    assert "Time Spent: 1h" in unit.content
    assert unit.metadata == {
        "language": "Python",
        "project": "my-app",
        "file": "src/main.py",
        "branch": "main",
        "editor": "VS Code",
        "time_spent_seconds": "3600",
        "time": "2025-01-15T10:00:00Z",
    }
    assert unit.tags == ["python", "my-app"]
    assert unit.created_at == datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)


def test_wakatime_csv_ingests_coding_sessions(tmp_path):
    export = tmp_path / "wakatime.csv"
    export.write_text(
        "\n".join(
            [
                "Date,Language,Project,File,Editor,Duration",
                "2025-01-20,JavaScript,web-project,src/app.js,WebStorm,1800",
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "JavaScript in web-project on 2025-01-20"
    assert unit.metadata["language"] == "JavaScript"
    assert unit.metadata["project"] == "web-project"
    assert unit.metadata["file"] == "src/app.js"
    assert unit.metadata["editor"] == "WebStorm"
    assert unit.tags == ["javascript", "web-project"]


def test_wakatime_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {
                    "language": "Go",
                    "time": "2025-01-22",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Go on 2025-01-22"
    assert unit.metadata["language"] == "Go"
    assert unit.metadata["project"] == ""
    assert unit.metadata["file"] == ""
    assert unit.tags == ["go"]


def test_wakatime_formats_duration_correctly(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"language": "Ruby", "duration": "45", "time": "2025-01-01"},  # 45 seconds
                {"language": "Java", "duration": "300", "time": "2025-01-02"},  # 5 minutes
                {"language": "C++", "duration": "7200", "time": "2025-01-03"},  # 2 hours
                {"language": "Rust", "duration": "5430", "time": "2025-01-04"},  # 1h 30m
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 4
    assert "Time Spent: 45s" in result.units[0].content
    assert "Time Spent: 5m" in result.units[1].content
    assert "Time Spent: 2h" in result.units[2].content
    assert "Time Spent: 1h 30m" in result.units[3].content


def test_wakatime_filters_system_files(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"language": "Python", "file": "node_modules/package/index.js", "time": "2025-01-01"},
                {"language": "Python", "file": ".git/config", "time": "2025-01-02"},
                {"language": "Python", "file": "__pycache__/module.pyc", "time": "2025-01-03"},
                {"language": "Python", "file": "src/main.py", "time": "2025-01-04"},  # Valid
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    # Only the valid file should be included
    assert len(result.units) == 1
    assert result.units[0].metadata["file"] == "src/main.py"


def test_wakatime_since_filters_sessions_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"language": "Python", "time": "2025-01-01"},
                {"language": "Python", "time": "2025-01-02"},
                {"language": "Python", "time": "2025-01-03"},
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="wakatime",
        source_entity_type="coding_session",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = WakaTimeAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert "2025-01-03" in result.units[0].title


def test_wakatime_respects_entity_types(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps([{"language": "Python", "project": "test"}]), encoding="utf-8"
    )

    result = WakaTimeAdapter(path=str(export)).ingest(entity_types=["activity"])

    assert result.units == []
    assert result.edges == []


def test_wakatime_skips_sessions_without_language_and_project(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"file": "test.txt", "time": "2025-01-01"},  # No language or project
                {"language": "Python", "time": "2025-01-02"},  # Valid
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["language"] == "Python"


def test_wakatime_handles_malformed_json(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text("This is not valid JSON {[", encoding="utf-8")

    result = WakaTimeAdapter(path=str(export)).ingest()

    # Should return empty result without crashing
    assert result.units == []


def test_wakatime_handles_empty_export(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(json.dumps([]), encoding="utf-8")

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert result.units == []


def test_wakatime_handles_nonexistent_file():
    result = WakaTimeAdapter(path="/nonexistent/path.json").ingest()

    assert result.units == []


def test_wakatime_creates_tags_from_language_and_project(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"language": "Type Script", "project": "my project", "time": "2025-01-01"},
                {"language": "Python", "time": "2025-01-02"},  # No project
                {"project": "solo-project", "time": "2025-01-03"},  # No language
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    assert result.units[0].tags == ["type-script", "my-project"]
    assert result.units[1].tags == ["python"]
    assert result.units[2].tags == ["solo-project"]


def test_wakatime_creates_unique_source_ids(tmp_path):
    export = tmp_path / "wakatime.json"
    export.write_text(
        json.dumps(
            [
                {"language": "Python", "project": "app", "time": "2025-01-01T10:00:00Z", "file": "main.py"},
                {"language": "Python", "project": "app", "time": "2025-01-01T11:00:00Z", "file": "main.py"},
            ]
        ),
        encoding="utf-8",
    )

    result = WakaTimeAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    # Source IDs should be different because they include time
    assert result.units[0].source_id != result.units[1].source_id


def test_wakatime_adapter_is_registered():
    assert "wakatime" in list_adapters()
    adapter = get_adapter("wakatime", path="/tmp/wakatime.json")
    assert adapter.name == "wakatime"
