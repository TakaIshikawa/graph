from __future__ import annotations

from datetime import datetime, timezone
import json

from graph.adapters.github_pull_requests_json import GithubPullRequestsJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_github_pull_requests_json_ingests_array_payload(tmp_path):
    export = tmp_path / "pulls.json"
    export.write_text(
        json.dumps(
            [
                {
                    "number": 42,
                    "title": "Add JSON adapter",
                    "body": "Adapter details",
                    "state": "open",
                    "user": {"login": "ada"},
                    "repository": {"full_name": "acme/graph"},
                    "html_url": "https://github.com/acme/graph/pull/42",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-02T03:04:05Z",
                    "merged_at": None,
                    "labels": [{"name": "enhancement"}, {"name": "Import"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GithubPullRequestsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITHUB_PULL_REQUESTS_JSON
    assert unit.source_id == "github_pull_requests_json:acme/graph#42"
    assert unit.source_entity_type == "pull_request"
    assert unit.metadata["number"] == 42
    assert unit.metadata["state"] == "open"
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["repository"] == "acme/graph"
    assert unit.metadata["url"] == "https://github.com/acme/graph/pull/42"
    assert unit.metadata["labels"] == ["enhancement", "import"]
    assert unit.metadata["body"] == "Adapter details"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.tags[:2] == ["github", "pull_request"]


def test_github_pull_requests_json_ingests_wrapped_nodes_payload(tmp_path):
    export = tmp_path / "wrapped.json"
    export.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "number": 7,
                        "title": "Ship GraphQL shape",
                        "bodyText": "Wrapped body",
                        "state": "MERGED",
                        "author": {"login": "grace"},
                        "repository": {"nameWithOwner": "acme/api"},
                        "url": "https://github.com/acme/api/pull/7",
                        "createdAt": "2025-01-03T00:00:00Z",
                        "updatedAt": "2025-01-04T00:00:00Z",
                        "mergedAt": "2025-01-05T00:00:00Z",
                        "labels": {"nodes": [{"name": "Backend"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = GithubPullRequestsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "github_pull_requests_json:acme/api#7"
    assert unit.metadata["state"] == "MERGED"
    assert unit.metadata["labels"] == ["backend"]
    assert unit.metadata["merged_at"] == "2025-01-05T00:00:00+00:00"
    assert "Wrapped body" in unit.content


def test_github_pull_requests_json_supports_pull_requests_wrapper_since_filtering_and_stable_ids(
    tmp_path,
):
    (tmp_path / "old.json").write_text(
        json.dumps(
            {
                "pull_requests": [
                    {
                        "title": "Old",
                        "url": "https://example.com/old",
                        "updated_at": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "new.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "title": "New",
                        "url": "https://example.com/new",
                        "updated_at": "2025-01-03T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = GithubPullRequestsJsonAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="github_pull_requests_json",
        source_entity_type="pull_request",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["issue"]).units == []


def test_github_pull_requests_json_skips_malformed_empty_and_bad_records(tmp_path):
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "empty.json").write_text("[]", encoding="utf-8")
    (tmp_path / "records.json").write_text(
        json.dumps([{"state": "open"}, "skip"]), encoding="utf-8"
    )

    result = GithubPullRequestsJsonAdapter(path=str(tmp_path)).ingest()

    assert result.units == []
    assert result.edges == []
