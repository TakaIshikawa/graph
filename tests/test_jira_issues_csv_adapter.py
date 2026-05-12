from __future__ import annotations

from graph.adapters.jira_issues_csv import JiraIssuesCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_jira_issues_csv_ingests_case_insensitive_headers_metadata_and_edges(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text(
        "\n".join(
            [
                "issue key,Summary,Description,Issue Type,Status,Priority,Assignee,Reporter,Created,Updated,Resolved,Labels,Components,Fix versions,Parent key",
                "PROJ-1,Add import,Details,Task,In Progress,High,Ada,Grace,2025-01-01,2025-01-02,2025-01-03,\"import,bug\",API,v1,PROJ-0",
            ]
        ),
        encoding="utf-8",
    )

    result = JiraIssuesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.JIRA_ISSUES_CSV
    assert unit.source_id == "jira_issues_csv:PROJ-1"
    assert unit.metadata["status"] == "In Progress"
    assert unit.metadata["issue_type"] == "Task"
    assert unit.metadata["priority"] == "High"
    assert unit.metadata["labels"] == ["import", "bug"]
    assert unit.metadata["components"] == ["API"]
    assert unit.metadata["fix_versions"] == ["v1"]
    assert "component:API" in unit.tags
    assert {edge.metadata["kind"] for edge in result.edges} == {"assignee", "reporter", "parent_key"}
    assert get_adapter("jira_issues_csv", path=str(export)).name == "jira_issues_csv"
