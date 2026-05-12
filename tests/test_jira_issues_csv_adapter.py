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

    issues = [unit for unit in result.units if unit.source_entity_type == "issue"]
    components = [unit for unit in result.units if unit.source_entity_type == "component"]
    assert len(issues) == 1
    assert len(components) == 1
    unit = issues[0]
    assert unit.source_project == SourceProject.JIRA_ISSUES_CSV
    assert unit.source_id == "jira_issues_csv:PROJ-1"
    assert unit.metadata["status"] == "In Progress"
    assert unit.metadata["issue_type"] == "Task"
    assert unit.metadata["priority"] == "High"
    assert unit.metadata["labels"] == ["import", "bug"]
    assert unit.metadata["components"] == ["API"]
    assert unit.metadata["fix_versions"] == ["v1"]
    assert "component:API" in unit.tags
    assert components[0].title == "API"
    assert (unit.source_id, components[0].source_id) in {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges}
    assert {edge.metadata["kind"] for edge in result.edges} == {"assignee", "reporter", "parent_key", "component"}
    assert get_adapter("jira_issues_csv", path=str(export)).name == "jira_issues_csv"


def test_jira_issues_csv_ingests_multiple_components_and_skips_missing(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text(
        "\n".join(
            [
                "Issue key,Summary,Components",
                "PROJ-1,One,\" API ; UI,API \"",
                "PROJ-2,Two,",
            ]
        ),
        encoding="utf-8",
    )

    result = JiraIssuesCsvAdapter(path=str(export)).ingest()

    components = [unit for unit in result.units if unit.source_entity_type == "component"]
    assert sorted(unit.title for unit in components) == ["API", "UI"]
    assert len([edge for edge in result.edges if edge.metadata["kind"] == "component"]) == 2
