from __future__ import annotations

import json

from graph.adapters.github_starred_repos_json import GitHubStarredReposJsonAdapter


def test_github_starred_repos_json_parses_array(tmp_path):
    path = tmp_path / "stars.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 123,
                    "full_name": "owner/repo",
                    "html_url": "https://github.com/owner/repo",
                    "description": "Useful project",
                    "language": "Python",
                    "topics": ["cli", "knowledge"],
                    "stargazers_count": 42,
                    "starred_at": "2026-01-02T03:04:05Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = GitHubStarredReposJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "github_starred_repos_json"
    assert unit.title == "owner/repo"
    assert unit.metadata["html_url"] == "https://github.com/owner/repo"
    assert unit.metadata["language"] == "Python"
    assert unit.metadata["topics"] == ["cli", "knowledge"]
    assert unit.metadata["stargazers_count"] == "42"
    assert unit.metadata["starred_at"] == "2026-01-02T03:04:05Z"
    assert unit.tags == ["cli", "knowledge"]
    assert "Useful project" in unit.content


def test_github_starred_repos_json_allows_missing_optional_fields(tmp_path):
    path = tmp_path / "stars.json"
    path.write_text(json.dumps([{"full_name": "owner/minimal"}]), encoding="utf-8")

    unit = GitHubStarredReposJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.title == "owner/minimal"
    assert unit.metadata["full_name"] == "owner/minimal"
    assert "language" not in unit.metadata
