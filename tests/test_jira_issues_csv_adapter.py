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
    assert {edge.metadata["kind"] for edge in result.edges} == {"assignee", "reporter", "parent_key", "component", "fix_version"}
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


def test_jira_issues_csv_emits_fix_version_aggregates_and_edges(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text(
        "\n".join(
            [
                "Issue key,Summary,Status,Components,Fix versions,Created,Updated",
                "PROJ-1,One,Done,API,\"v1, v2\",2025-01-01,2025-01-03",
                "PROJ-2,Two,In Progress,UI,V1,2025-01-02,2025-01-04",
                "PROJ-3,Three,Todo,, ,2025-01-05,2025-01-06",
            ]
        ),
        encoding="utf-8",
    )

    result = JiraIssuesCsvAdapter(path=str(export)).ingest(entity_types=["issue", "fix_version"])

    assert "fix_version" in JiraIssuesCsvAdapter(path=str(export)).entity_types
    versions = {unit.title: unit for unit in result.units if unit.source_entity_type == "fix_version"}
    assert set(versions) == {"v1", "v2"}
    assert versions["v1"].metadata["issue_count"] == 2
    assert versions["v1"].metadata["statuses"] == ["Done", "In Progress"]
    assert versions["v1"].metadata["components"] == ["API", "UI"]
    assert versions["v1"].metadata["first_created_at"] == "2025-01-01T00:00:00+00:00"
    assert versions["v1"].metadata["last_updated_at"] == "2025-01-04T00:00:00+00:00"
    assert versions["v1"].metadata["issue_source_ids"] == ["jira_issues_csv:PROJ-1", "jira_issues_csv:PROJ-2"]

    edges = [edge for edge in result.edges if edge.metadata["kind"] == "fix_version"]
    assert len(edges) == 3
    assert {edge.relation for edge in edges} == {EdgeRelation.RELATES_TO}
    assert {edge.to_unit_id for edge in edges} == {unit.source_id for unit in versions.values()}


def test_jira_issues_csv_fix_version_entity_filtering(tmp_path):
    export = tmp_path / "jira.csv"
    export.write_text("Issue key,Summary,Fix versions\nPROJ-1,One,v1\n", encoding="utf-8")

    versions = JiraIssuesCsvAdapter(path=str(export)).ingest(entity_types=["fix_version"])

    assert [unit.source_entity_type for unit in versions.units] == ["fix_version"]
    assert versions.edges == []
