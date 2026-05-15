from __future__ import annotations

from graph.adapters.jira_worklogs_csv import JiraWorklogsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_jira_worklogs_csv_ingests_worklog_rows(tmp_path):
    export = tmp_path / "worklogs.csv"
    export.write_text(
        "Worklog ID,Issue key,Issue summary,Author,Started,Time spent,Comment,Project,URL\n"
        "10001,PROJ-1,Add importer,Ada,2026-05-01T09:30:00Z,1h 30m,Implemented parser,PROJ,https://jira.example/browse/PROJ-1\n",
        encoding="utf-8",
    )

    result = JiraWorklogsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.JIRA_WORKLOGS_CSV
    assert unit.source_id == "jira_worklogs_csv:10001"
    assert unit.source_entity_type == "worklog"
    assert unit.metadata["issue_key"] == "PROJ-1"
    assert unit.metadata["issue_summary"] == "Add importer"
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["started_at"] == "2026-05-01T09:30:00+00:00"
    assert unit.metadata["time_spent"] == "1h 30m"
    assert unit.metadata["time_spent_seconds"] == 5400
    assert unit.metadata["comment"] == "Implemented parser"
    assert unit.metadata["project"] == "PROJ"
    assert unit.metadata["source_url"] == "https://jira.example/browse/PROJ-1"
    assert "Comment: Implemented parser" in unit.content


def test_jira_worklogs_csv_tolerates_blank_comment_and_optional_columns(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text(
        "Issue key,Summary,Author,Started,Seconds,Comment\n"
        "PROJ-2,Review import,Grace,2026-05-02,1800,\n",
        encoding="utf-8",
    )

    result = JiraWorklogsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["issue_key"] == "PROJ-2"
    assert unit.metadata["issue_summary"] == "Review import"
    assert unit.metadata["time_spent_seconds"] == 1800
    assert "comment" not in unit.metadata
    assert "project" not in unit.metadata
    assert "source_url" not in unit.metadata


def test_jira_worklogs_csv_is_registered():
    assert "jira_worklogs_csv" in list_adapters()
    assert isinstance(get_adapter("jira-worklogs-csv"), JiraWorklogsCsvAdapter)
