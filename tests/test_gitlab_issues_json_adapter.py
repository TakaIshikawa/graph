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
