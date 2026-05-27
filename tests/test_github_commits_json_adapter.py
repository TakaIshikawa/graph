from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_commits_json import GithubCommitsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_github_commits_json_ingests_json_array_with_sha_external_id(tmp_path):
    export = tmp_path / "commits.json"
    export.write_text(
        json.dumps(
            [
                {
                    "sha": "abc123",
                    "html_url": "https://github.com/acme/graph/commit/abc123",
                    "commit": {
                        "message": "Add adapter\n\nPreserve GitHub commit details.",
                        "author": {"name": "Ada Lovelace", "date": "2025-01-01T10:00:00Z"},
                        "committer": {"name": "Grace Hopper", "date": "2025-01-01T11:00:00Z"},
                    },
                    "parents": [{"sha": "parent1"}, {"sha": "parent2"}],
                    "stats": {"additions": 5, "deletions": 2, "total": 7},
                    "files": [
                        {"filename": "src/graph/adapters/github_commits_json.py", "status": "added"},
                        {"filename": "tests/test_github_commits_json_adapter.py", "status": "added"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GithubCommitsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITHUB_COMMITS_JSON
    assert unit.source_id == "github_commits_json:abc123"
    assert unit.metadata["external_id"] == "abc123"
    assert unit.title == "Add adapter"
    assert unit.content == "Add adapter\n\nPreserve GitHub commit details."
    assert unit.metadata["html_url"] == "https://github.com/acme/graph/commit/abc123"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["committer"] == "Grace Hopper"
    assert unit.metadata["parents"] == ["parent1", "parent2"]
    assert unit.metadata["filenames"] == [
        "src/graph/adapters/github_commits_json.py",
        "tests/test_github_commits_json_adapter.py",
    ]
    assert unit.metadata["stats"] == {"additions": 5, "deletions": 2, "total": 7}
    assert unit.updated_at == datetime(2025, 1, 1, 11, tzinfo=timezone.utc)


def test_github_commits_json_accepts_commits_wrapper_and_registry(tmp_path):
    export = tmp_path / "wrapped.json"
    export.write_text(
        json.dumps(
            {
                "commits": [
                    {
                        "sha": "def456",
                        "commit": {
                            "message": "Fix wrapped export",
                            "author": {"name": "Ada", "date": "2025-02-01T00:00:00Z"},
                        },
                        "files": ["README.md"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = get_adapter("github_commits_json", path=str(export)).ingest()

    assert get_adapter("github_commits_json").name == "github_commits_json"
    assert len(result.units) == 1
    assert result.units[0].source_id == "github_commits_json:def456"
    assert result.units[0].metadata["files"] == [{"filename": "README.md"}]
