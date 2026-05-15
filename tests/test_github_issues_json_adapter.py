from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_issues_json import GithubIssuesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.enums import SourceProject


def test_github_issues_json_ingests_json_array(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 42,
                    "title": "Fix import",
                    "body": "Details",
                    "state": "open",
                    "labels": [{"name": "bug"}, {"name": "Import"}],
                    "repository": {"full_name": "acme/graph"},
                    "user": {"login": "ada"},
                    "html_url": "https://github.com/acme/graph/issues/42",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GithubIssuesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITHUB_ISSUES_JSON
    assert unit.source_id == "github_issues_json:acme/graph#42"
    assert unit.source_entity_type == "issue"
    assert unit.metadata["state"] == "open"
    assert unit.metadata["labels"] == ["bug", "import"]
    assert unit.metadata["repository"] == "acme/graph"
    assert unit.metadata["author"] == "ada"
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "bug" in unit.tags


def test_github_issues_json_ingests_jsonl_pull_requests_and_registry(tmp_path):
    export = tmp_path / "issues.jsonl"
    export.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "number": 7,
                        "title": "Add adapter",
                        "state": "closed",
                        "labels": "enhancement,imports",
                        "repository_full_name": "acme/graph",
                        "html_url": "https://github.com/acme/graph/pull/7",
                        "pull_request": {"merged_at": "2025-01-03T00:00:00Z"},
                        "created_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-03T00:00:00Z",
                        "closed_at": "2025-01-03T00:00:00Z",
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    result = GithubIssuesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "pull_request"
    assert unit.metadata["pull_request"]["merged_at"] == "2025-01-03T00:00:00Z"
    assert unit.metadata["closed_at"] == "2025-01-03T00:00:00+00:00"
    assert unit.tags[:2] == ["github", "pull_request"]
    assert get_adapter("github_issues_json", path=str(export)).name == "github_issues_json"


def test_github_issues_json_preserves_milestone_metadata(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 8,
                    "title": "Ship milestone",
                    "repository_full_name": "acme/graph",
                    "milestone": {
                        "title": "v1.0",
                        "state": "open",
                        "due_on": "2025-02-01T00:00:00Z",
                        "number": 3,
                    },
                },
                {
                    "number": 9,
                    "title": "No milestone",
                    "repository_full_name": "acme/graph",
                    "milestone": None,
                },
            ]
        ),
        encoding="utf-8",
    )

    units = {unit.title: unit for unit in GithubIssuesJsonAdapter(path=str(export)).ingest().units}

    assert units["Ship milestone"].metadata["milestone_title"] == "v1.0"
    assert units["Ship milestone"].metadata["milestone_state"] == "open"
    assert units["Ship milestone"].metadata["milestone_due_on"] == "2025-02-01T00:00:00+00:00"
    assert units["Ship milestone"].metadata["milestone_number"] == 3
    assert "v1.0" in units["Ship milestone"].tags
    assert "milestone_title" not in units["No milestone"].metadata


def test_github_issues_json_emits_relationship_edges_deterministically(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 10,
                    "title": "Link relationships",
                    "body": "See https://example.com/spec and https://example.com/spec.",
                    "repository_full_name": "acme/graph",
                    "user": {"login": "ada"},
                    "assignees": [{"login": "grace"}, {"login": "grace"}],
                    "milestone": {"title": "v1.0"},
                },
                {
                    "number": 11,
                    "title": "Single assignee",
                    "body": "See https://example.com/other.",
                    "repository_full_name": "acme/graph",
                    "assignee": {"login": "grace"},
                },
            ]
        ),
        encoding="utf-8",
    )

    result = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["issue", "milestone"])

    assert len([unit for unit in result.units if unit.source_entity_type == "issue"]) == 2
    assert len(result.edges) == len({edge.id for edge in result.edges})
    first_edges = [edge for edge in result.edges if edge.from_unit_id == "github_issues_json:acme/graph#10"]
    assert [edge.metadata["kind"] for edge in first_edges] == ["assignee", "author", "mentioned_url", "milestone"]
    assert {edge.metadata["value"] for edge in first_edges} == {"ada", "grace", "v1.0", "https://example.com/spec"}
    assert {edge.relation for edge in first_edges} == {EdgeRelation.RELATES_TO, EdgeRelation.REFERENCES}
    assert all(edge.source == EdgeSource.SOURCE for edge in first_edges)


def test_github_issues_json_ingests_label_aggregates_and_edges(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 1,
                    "title": "Fix bug",
                    "labels": [{"name": "Bug"}, {"name": "Import"}],
                    "repository_full_name": "acme/graph",
                    "updated_at": "2025-01-02T00:00:00Z",
                },
                {
                    "number": 2,
                    "title": "Ship PR",
                    "labels": ["bug"],
                    "repository_full_name": "acme/graph",
                    "pull_request": {},
                    "updated_at": "2025-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    labels_only = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["label"])
    combined = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["issue", "pull_request", "label"])

    assert GithubIssuesJsonAdapter().entity_types == ["issue", "pull_request", "label", "milestone"]
    labels = {unit.metadata["label"]: unit for unit in labels_only.units}
    assert sorted(labels) == ["bug", "import"]
    bug = labels["bug"]
    assert bug.source_entity_type == "label"
    assert bug.metadata["issue_source_ids"] == [
        "github_issues_json:acme/graph#1",
        "github_issues_json:acme/graph#2",
    ]
    assert bug.metadata["issue_count"] == 2
    assert labels_only.edges == []

    label_edges = [edge for edge in combined.edges if edge.metadata["kind"] == "label"]
    assert len(label_edges) == 3
    assert {edge.relation for edge in label_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in label_edges} == {EdgeSource.SOURCE}


def test_github_issues_json_emits_milestone_aggregates_and_edges(tmp_path):
    export = tmp_path / "issues.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 1,
                    "title": "Fix bug",
                    "repository_full_name": "acme/graph",
                    "milestone": {
                        "title": "v1.0",
                        "state": "open",
                        "due_on": "2025-02-01T00:00:00Z",
                        "number": 3,
                    },
                    "updated_at": "2025-01-02T00:00:00Z",
                },
                {
                    "number": 2,
                    "title": "Ship PR",
                    "repository_full_name": "acme/graph",
                    "pull_request": {},
                    "milestone": {"title": "v1.0", "state": "open", "number": 3},
                    "updated_at": "2025-01-03T00:00:00Z",
                },
                {
                    "number": 3,
                    "title": "Next release",
                    "repository_full_name": "acme/graph",
                    "milestone": {"title": "v2.0", "state": "closed", "number": 4},
                    "updated_at": "2025-01-04T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    milestones_only = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["milestone"])
    combined = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["issue", "pull_request", "milestone"])
    issues_only = GithubIssuesJsonAdapter(path=str(export)).ingest(entity_types=["issue", "pull_request"])

    assert GithubIssuesJsonAdapter().entity_types == ["issue", "pull_request", "label", "milestone"]
    milestones = {unit.metadata["milestone_title"]: unit for unit in milestones_only.units}
    assert sorted(milestones) == ["v1.0", "v2.0"]
    assert milestones["v1.0"].source_entity_type == "milestone"
    assert milestones["v1.0"].source_id.startswith("github_issues_json:milestone:")
    assert milestones["v1.0"].metadata["milestone_state"] == "open"
    assert milestones["v1.0"].metadata["milestone_due_on"] == "2025-02-01T00:00:00+00:00"
    assert milestones["v1.0"].metadata["milestone_number"] == 3
    assert milestones["v1.0"].metadata["issue_source_ids"] == [
        "github_issues_json:acme/graph#1",
        "github_issues_json:acme/graph#2",
    ]
    assert milestones["v1.0"].metadata["issue_count"] == 2
    assert milestones["v1.0"].metadata["repositories"] == ["acme/graph"]
    assert milestones["v1.0"].metadata["states"] == ["open"]
    assert milestones["v1.0"].metadata["due_dates"] == ["2025-02-01T00:00:00+00:00"]
    assert milestones["v1.0"].metadata["numbers"] == [3]
    assert milestones_only.edges == []

    milestone_edges = [edge for edge in combined.edges if edge.metadata["kind"] == "milestone"]
    assert len(milestone_edges) == 3
    assert {edge.to_unit_id for edge in milestone_edges} == {unit.source_id for unit in combined.units if unit.source_entity_type == "milestone"}
    assert {edge.relation for edge in milestone_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in milestone_edges} == {EdgeSource.SOURCE}
    assert all(edge.metadata["to_entity_type"] == "milestone" for edge in milestone_edges)
    assert any(edge.metadata.get("milestone_due_on") == "2025-02-01T00:00:00+00:00" for edge in milestone_edges)

    assert {unit.source_entity_type for unit in issues_only.units} == {"issue", "pull_request"}
    assert all(edge.metadata["kind"] != "milestone" for edge in issues_only.edges)
