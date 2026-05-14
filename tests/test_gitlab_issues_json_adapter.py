from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.gitlab_issues_json import GitlabIssuesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


def test_gitlab_issues_json_ingests_top_level_list_with_relationship_edges(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "iid": 42,
                    "title": "Fix import",
                    "description": "See https://example.com/spec",
                    "state": "opened",
                    "labels": ["Bug", "Import"],
                    "project": {"path_with_namespace": "acme/graph"},
                    "author": {"username": "ada"},
                    "assignees": [{"username": "grace"}],
                    "milestone": {"title": "v1"},
                    "web_url": "https://gitlab.com/acme/graph/-/issues/42",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z",
                    "closed_at": "2025-01-03T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GitlabIssuesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITLAB_ISSUES_JSON
    assert unit.source_id == "gitlab_issues_json:acme/graph#42"
    assert unit.metadata["body"] == "See https://example.com/spec"
    assert unit.metadata["labels"] == ["bug", "import"]
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["assignees"] == ["grace"]
    assert unit.metadata["milestone"] == "v1"
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "bug" in unit.tags
    edge_targets = {edge.to_unit_id for edge in result.edges}
    assert "gitlab:author:ada" in edge_targets
    assert "gitlab:assignee:grace" in edge_targets
    assert "gitlab:milestone:v1" in edge_targets
    assert "url:https://example.com/spec" in edge_targets
    assert any(edge.relation == EdgeRelation.REFERENCES for edge in result.edges)


def test_gitlab_issues_json_ingests_object_with_issues_and_registry(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "title": "URL only",
                        "body": "No project path",
                        "labels": "ops,team",
                        "web_url": "https://gitlab.example.com/team/project/-/issues/9",
                        "created_at": "2025-02-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GitlabIssuesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("gitlab_issues_json:")
    assert result.units[0].metadata["labels"] == ["ops", "team"]
    assert get_adapter("gitlab_issues_json", path=str(export)).name == "gitlab_issues_json"


def test_gitlab_issues_json_ingests_label_aggregates_and_edges(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "iid": 1,
                    "title": "Fix import",
                    "labels": ["Bug", "Import"],
                    "project_path": "acme/graph",
                    "updated_at": "2025-01-02T00:00:00Z",
                },
                {
                    "iid": 2,
                    "title": "Fix another import",
                    "labels": "bug",
                    "project_path": "acme/graph",
                    "updated_at": "2025-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    labels_only = GitlabIssuesJsonAdapter(path=str(export)).ingest(entity_types=["label"])
    combined = GitlabIssuesJsonAdapter(path=str(export)).ingest(entity_types=["issue", "label"])

    assert GitlabIssuesJsonAdapter().entity_types == ["issue", "label"]
    labels = {unit.metadata["label"]: unit for unit in labels_only.units}
    assert sorted(labels) == ["bug", "import"]
    assert labels["bug"].source_id.startswith("gitlab_issues_json:label:")
    assert labels["bug"].metadata["issue_source_ids"] == [
        "gitlab_issues_json:acme/graph#1",
        "gitlab_issues_json:acme/graph#2",
    ]
    assert labels["bug"].metadata["issue_count"] == 2
    assert labels["bug"].metadata["project_paths"] == ["acme/graph"]
    assert labels["bug"].metadata["latest_updated_at"] == "2025-01-03T00:00:00+00:00"
    assert labels_only.edges == []

    label_edges = [edge for edge in combined.edges if edge.metadata["kind"] == "label"]
    assert len(label_edges) == 3
    assert {edge.relation for edge in label_edges} == {EdgeRelation.RELATES_TO}
