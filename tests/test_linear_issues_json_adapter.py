from __future__ import annotations

import json

from graph.adapters.linear_issues_json import LinearIssuesJsonAdapter
from graph.adapters.registry import get_adapter


def test_linear_issues_json_ingests_wrapped_export_and_relationships(tmp_path):
    export = tmp_path / "linear.json"
    export.write_text(json.dumps({"issues": [{"id": "p", "identifier": "LIN-1", "title": "Parent", "state": {"name": "Open"}, "labels": [{"name": "import"}], "createdAt": "2026-05-01T00:00:00Z"}, {"id": "c", "identifier": "LIN-2", "title": "Child", "parent": {"id": "p"}, "relatedIssueIds": ["p"], "createdAt": "2026-05-02T00:00:00Z"}]}), encoding="utf-8")

    result = LinearIssuesJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["linear_issues_json:c", "linear_issues_json:p"]
    assert result.units[1].metadata["labels"] == ["import"]
    assert {edge.metadata["kind"] for edge in result.edges} == {"parent", "related"}
    assert get_adapter("linear_issues_json", path=str(export)).name == "linear_issues_json"


def test_linear_issues_json_adds_lifecycle_day_metadata(tmp_path):
    export = tmp_path / "linear.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "life",
                    "identifier": "LIN-3",
                    "title": "Lifecycle issue",
                    "createdAt": "2026-05-01T00:00:00Z",
                    "updatedAt": "2026-05-12T00:00:00Z",
                    "triagedAt": "2026-05-02T00:00:00Z",
                    "startedAt": "2026-05-04T00:00:00Z",
                    "completedAt": "2026-05-10T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = LinearIssuesJsonAdapter(path=str(export)).ingest().units[0].metadata

    assert metadata["triaged_at"] == "2026-05-02T00:00:00+00:00"
    assert metadata["started_at"] == "2026-05-04T00:00:00+00:00"
    assert metadata["completed_at"] == "2026-05-10T00:00:00+00:00"
    assert metadata["age_days"] == 11
    assert metadata["time_to_triage_days"] == 1
    assert metadata["time_to_start_days"] == 3
    assert metadata["cycle_time_days"] == 6
    assert metadata["lead_time_days"] == 9
    assert metadata["terminal_state_age_days"] == 2


def test_linear_issues_json_ignores_malformed_lifecycle_timestamps(tmp_path):
    export = tmp_path / "linear.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "bad-life",
                    "identifier": "LIN-4",
                    "title": "Malformed lifecycle issue",
                    "created_at": "2026-05-01",
                    "updated_at": "2026-05-03",
                    "started_at": "not a date",
                    "canceled_at": "also bad",
                    "archived_at": "2026-05-02",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = LinearIssuesJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Malformed lifecycle issue"
    assert "started_at" not in unit.metadata
    assert "canceled_at" not in unit.metadata
    assert unit.metadata["archived_at"] == "2026-05-02T00:00:00+00:00"
    assert unit.metadata["lead_time_days"] == 1
    assert unit.metadata["terminal_state_age_days"] == 1
