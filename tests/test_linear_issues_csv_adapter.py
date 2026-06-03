from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.linear_issues_csv import LinearIssuesCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_linear_issues_csv_ingests_labels_blank_assignee_and_stable_ids(tmp_path):
    export = tmp_path / "linear.csv"
    export.write_text(
        "\n".join(
            [
                "Identifier,Title,Description,Status,Priority,Assignee,Team,Project,Labels,Created,Updated,URL",
                'ENG-1,Add importer,Details,In Progress,High,,Platform,Graph,"import,backend",2025-01-01,2025-01-02,https://linear.app/acme/issue/ENG-1',
            ]
        ),
        encoding="utf-8",
    )

    result = LinearIssuesCsvAdapter(path=str(export)).ingest()
    again = LinearIssuesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.LINEAR_ISSUES_CSV
    assert unit.source_id == "linear_issues_csv:ENG-1"
    assert unit.source_id == again.units[0].source_id
    assert unit.metadata["labels"] == ["import", "backend"]
    assert "assignee" not in unit.metadata
    assert unit.metadata["team"] == "Platform"
    assert unit.metadata["project"] == "Graph"
    assert unit.metadata["url"] == "https://linear.app/acme/issue/ENG-1"
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert get_adapter("linear_issues_csv", path=str(export)).name == "linear_issues_csv"


def test_linear_issues_csv_accepts_done_archived_and_since_filter(tmp_path):
    export = tmp_path / "linear.csv"
    export.write_text(
        "\n".join(
            [
                "Identifier,Title,Status,Completed At,Archived At,Updated",
                "ENG-1,Old,Done,2025-01-01,2025-01-02,2025-01-01",
                "ENG-2,New,Archived,,2025-01-04,2025-01-04",
            ]
        ),
        encoding="utf-8",
    )

    sync = SyncState(source_project="linear_issues_csv", source_entity_type="issue", last_sync_at=datetime(2025, 1, 3, tzinfo=timezone.utc))
    result = LinearIssuesCsvAdapter(path=str(export)).ingest(since=sync)

    assert [unit.title for unit in result.units] == ["New"]
    assert result.units[0].metadata["archived_at"] == "2025-01-04T00:00:00+00:00"
    assert LinearIssuesCsvAdapter(path=str(export)).ingest(entity_types=["project"]).units == []
