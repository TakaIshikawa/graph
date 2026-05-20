from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.linear_cycles_json import LinearCyclesJsonAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_linear_cycles_json_ingests_wrapped_cycles(tmp_path):
    export = tmp_path / "cycles.json"
    export.write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "id": "cyc-1",
                        "number": 42,
                        "name": "May Iteration",
                        "team": {"key": "ENG", "name": "Engineering"},
                        "startsAt": "2026-05-01T00:00:00Z",
                        "endsAt": "2026-05-14T23:59:00Z",
                        "completedAt": "2026-05-15T10:00:00Z",
                        "progress": 0.75,
                        "issueCount": 20,
                        "completedIssueCount": 15,
                        "startedIssueCount": 3,
                        "uncompletedIssueCount": 5,
                        "scope": 60,
                        "completedScope": 45,
                        "startedScope": 9,
                        "description": "Stabilize imports",
                        "isArchived": False,
                        "url": "https://linear.app/acme/cycle/ENG-42",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = LinearCyclesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "linear_cycles_json"
    assert unit.source_id == "linear_cycles_json:cyc-1"
    assert unit.source_entity_type == "cycle"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "ENG May Iteration"
    assert unit.metadata["cycle_id"] == "cyc-1"
    assert unit.metadata["number"] == 42
    assert unit.metadata["name"] == "May Iteration"
    assert unit.metadata["team_key"] == "ENG"
    assert unit.metadata["team_name"] == "Engineering"
    assert unit.metadata["starts_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["ends_at"] == "2026-05-14T23:59:00+00:00"
    assert unit.metadata["completed_at"] == "2026-05-15T10:00:00+00:00"
    assert unit.metadata["progress"] == 0.75
    assert unit.metadata["issue_count"] == 20
    assert unit.metadata["completed_issue_count"] == 15
    assert unit.metadata["started_issue_count"] == 3
    assert unit.metadata["uncompleted_issue_count"] == 5
    assert unit.metadata["scope"] == 60.0
    assert unit.metadata["completed_scope"] == 45.0
    assert unit.metadata["started_scope"] == 9.0
    assert unit.metadata["description"] == "Stabilize imports"
    assert unit.metadata["archived"] is False
    assert unit.metadata["url"] == "https://linear.app/acme/cycle/ENG-42"
    assert unit.metadata["source_file"] == "cycles.json"
    assert {"linear", "cycle", "ENG", "Engineering"}.issubset(set(unit.tags))
    assert unit.updated_at == datetime(2026, 5, 15, 10, tzinfo=timezone.utc)
    assert "Issues: 20" in unit.content


def test_linear_cycles_json_ingests_api_nodes_and_uses_team_number_ids(tmp_path):
    export = tmp_path / "api.json"
    export.write_text(
        json.dumps(
            {
                "data": {
                    "cycles": {
                        "nodes": [
                            {
                                "number": "7",
                                "team": {"key": "OPS", "name": "Operations"},
                                "starts_at": "2026-04-01",
                                "ends_at": "2026-04-15",
                                "issue_count": "4",
                            },
                            {"id": "cyc-2", "number": 8, "teamKey": "OPS", "updatedAt": "2026-04-20T00:00:00Z"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    first = LinearCyclesJsonAdapter(path=str(export)).ingest().units
    second = LinearCyclesJsonAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert first[0].source_id.startswith("linear_cycles_json:")
    assert first[0].metadata["team_key"] == "OPS"
    assert first[0].metadata["number"] == 7
    assert first[0].metadata["issue_count"] == 4
    assert first[1].source_id == "linear_cycles_json:cyc-2"


def test_linear_cycles_json_directory_invalid_json_ordering_since_and_entity_filter(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps([{"id": "old", "teamKey": "ENG", "number": 1, "updatedAt": "2026-05-01T00:00:00Z"}]), encoding="utf-8")
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"cycles": [{"id": "new", "teamKey": "ENG", "number": 2, "updatedAt": "2026-05-03T00:00:00Z"}]}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    ignored = tmp_path / "notes.txt"
    ignored.write_text(json.dumps([{"id": "ignored"}]), encoding="utf-8")
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps([{"description": "No identity"}]), encoding="utf-8")
    since = SyncState(source_project="linear_cycles_json", source_entity_type="cycle", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = LinearCyclesJsonAdapter(path=str(tmp_path)).ingest(since=since)

    assert LinearCyclesJsonAdapter().entity_types == ["cycle"]
    assert [unit.source_id for unit in result.units] == ["linear_cycles_json:new"]
    all_units = LinearCyclesJsonAdapter(path=str(tmp_path)).ingest().units
    assert [unit.source_id for unit in all_units] == ["linear_cycles_json:new", "linear_cycles_json:old"]
    assert LinearCyclesJsonAdapter(path=str(tmp_path)).ingest(entity_types=["issue"]).units == []
