from __future__ import annotations

from graph.adapters.jira_issues_csv import JiraIssuesCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


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


def test_jira_issues_csv_creates_issue_link_edges(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text(
        "\n".join(
            [
                "Issue key,Summary,Blocks,Is blocked by,Relates to,Duplicates,Epic Link",
                "PROJ-1,One,\"PROJ-2; proj-3\",PROJ-4,\"PROJ-5 PROJ-5\",PROJ-6,EPIC-1",
                "PROJ-2,Two,,,,,",
            ]
        ),
        encoding="utf-8",
    )

    first = JiraIssuesCsvAdapter(path=str(export)).ingest()
    second = JiraIssuesCsvAdapter(path=str(export)).ingest()

    link_edges = [
        edge
        for edge in first.edges
        if edge.from_unit_id == "jira_issues_csv:PROJ-1"
        and edge.to_unit_id.startswith("jira_issues_csv:")
    ]
    assert sorted(
        (edge.to_unit_id, edge.relation, edge.metadata["kind"], edge.metadata["value"])
        for edge in link_edges
    ) == [
        ("jira_issues_csv:EPIC-1", EdgeRelation.REFERENCES, "epic", "EPIC-1"),
        ("jira_issues_csv:PROJ-2", EdgeRelation.RELATES_TO, "blocks", "PROJ-2"),
        ("jira_issues_csv:PROJ-3", EdgeRelation.RELATES_TO, "blocks", "PROJ-3"),
        ("jira_issues_csv:PROJ-4", EdgeRelation.RELATES_TO, "is_blocked_by", "PROJ-4"),
        ("jira_issues_csv:PROJ-5", EdgeRelation.RELATES_TO, "relates_to", "PROJ-5"),
        ("jira_issues_csv:PROJ-6", EdgeRelation.RELATES_TO, "duplicates", "PROJ-6"),
    ]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


def test_jira_issues_csv_empty_link_fields_do_not_create_issue_link_edges(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text(
        "\n".join(
            [
                "Issue key,Summary,Blocks,Relates to",
                "PROJ-1,One,,",
            ]
        ),
        encoding="utf-8",
    )

    result = JiraIssuesCsvAdapter(path=str(export)).ingest()

    assert result.edges == []
