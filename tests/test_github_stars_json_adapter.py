from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_stars_json import GithubStarsJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_github_stars_json_parses_list_shape(tmp_path):
    path = tmp_path / "stars.json"
    path.write_text(
        json.dumps(
            [
                {
                    "full_name": "owner/repo",
                    "owner": {"login": "owner"},
                    "name": "repo",
                    "html_url": "https://github.com/owner/repo",
                    "description": "Useful project",
                    "language": "Python",
                    "topics": ["cli", "knowledge"],
                    "starred_at": "2025-01-02T03:04:05Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = GithubStarsJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.GITHUB_STARS_CSV
    assert unit.source_entity_type == "repository"
    assert unit.title == "owner/repo"
    assert unit.content == "owner/repo\nUseful project\nLanguage: Python\nTopics: cli, knowledge\nURL: https://github.com/owner/repo"
    assert unit.metadata == {
        "full_name": "owner/repo",
        "owner": "owner",
        "repo": "repo",
        "html_url": "https://github.com/owner/repo",
        "language": "Python",
        "topics": ["cli", "knowledge"],
        "starred_at": "2025-01-02T03:04:05Z",
        "description": "Useful project",
        "source_file": str(path),
        "record_index": 0,
    }
    assert unit.tags == ["cli", "knowledge"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_github_stars_json_parses_wrapped_shapes_and_nested_repo(tmp_path):
    path = tmp_path / "stars.json"
    path.write_text(
        json.dumps(
            {
                "stars": [
                    {
                        "starredAt": "2025-01-03T00:00:00Z",
                        "repo": {
                            "owner": "acme",
                            "name": "widget",
                            "url": "https://github.com/acme/widget",
                            "description": "Widgets",
                            "primaryLanguage": "Go",
                            "repositoryTopics": {"nodes": [{"topic": {"name": "tools"}}]},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = GithubStarsJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["full_name"] == "acme/widget"
    assert unit.metadata["owner"] == "acme"
    assert unit.metadata["repo"] == "widget"
    assert unit.metadata["topics"] == ["tools"]
    assert unit.created_at == datetime(2025, 1, 3, tzinfo=timezone.utc)


def test_github_stars_json_since_entity_filter_and_stable_ids(tmp_path):
    path = tmp_path / "stars.json"
    path.write_text(
        json.dumps(
            {
                "repositories": [
                    {"full_name": "owner/old", "starred_at": "2025-01-01T00:00:00Z"},
                    {"full_name": "owner/new", "starred_at": "2025-01-03T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="github_stars_csv",
        source_entity_type="repository",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    filtered = GithubStarsJsonAdapter(path=str(path)).ingest(since=since)
    first = GithubStarsJsonAdapter(path=str(path)).ingest().units
    second = GithubStarsJsonAdapter(path=str(path)).ingest().units
    wrong_entity = GithubStarsJsonAdapter(path=str(path)).ingest(entity_types=["owner"])

    assert [unit.metadata["full_name"] for unit in filtered.units] == ["owner/new"]
    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert wrong_entity.units == []
