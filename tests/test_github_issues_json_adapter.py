from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_issues_json import GithubIssuesJsonAdapter
from graph.adapters.registry import get_adapter
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
