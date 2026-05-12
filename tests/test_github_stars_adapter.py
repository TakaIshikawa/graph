from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.github_stars_csv import GithubStarsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


def test_github_stars_csv_imports_repositories(tmp_path):
    path = tmp_path / "stars.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["full_name", "description", "html_url", "language", "stargazers_count", "topics", "owner", "starred_at"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "full_name": "owner/repo",
                "description": "Useful project",
                "html_url": "https://github.com/owner/repo",
                "language": "Python",
                "stargazers_count": "42",
                "topics": '["cli", "knowledge"]',
                "owner": "owner",
                "starred_at": "2025-01-02T03:04:05Z",
            }
        )

    unit = GithubStarsCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.GITHUB_STARS_CSV
    assert unit.title == "owner/repo"
    assert unit.content == "owner/repo\nUseful project\nURL: https://github.com/owner/repo\nTopics: cli, knowledge"
    assert unit.metadata["source_url"] == "https://github.com/owner/repo"
    assert unit.metadata["external_url"] == "https://github.com/owner/repo"
    assert unit.metadata["language"] == "Python"
    assert unit.metadata["topics"] == ["cli", "knowledge"]
    assert unit.metadata["owner"] == "owner"
    assert unit.metadata["stargazers_count"] == 42
    assert unit.metadata["starred_at"] == "2025-01-02T03:04:05Z"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_github_stars_csv_keeps_missing_description_url_units_and_topics(tmp_path):
    path = tmp_path / "stars.csv"
    path.write_text(
        "full_name,description,html_url,topics\nowner/repo,,https://github.com/owner/repo,\"python, testing\"\n",
        encoding="utf-8",
    )

    unit = GithubStarsCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.content.startswith("owner/repo\nhttps://github.com/owner/repo")
    assert unit.tags == ["python", "testing"]


def test_github_stars_csv_emits_owner_aggregates_and_edges(tmp_path):
    path = tmp_path / "stars.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "full_name",
                "description",
                "html_url",
                "language",
                "stargazers_count",
                "topics",
                "owner",
                "starred_at",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "full_name": "owner/alpha",
                    "description": "Alpha",
                    "html_url": "https://github.com/owner/alpha",
                    "language": "Python",
                    "stargazers_count": "10",
                    "topics": "cli,knowledge",
                    "owner": "Owner",
                    "starred_at": "2025-01-02T00:00:00Z",
                },
                {
                    "full_name": "owner/beta",
                    "description": "Beta",
                    "html_url": "https://github.com/owner/beta",
                    "language": "Go",
                    "stargazers_count": "15",
                    "topics": '["knowledge", "server"]',
                    "owner": "owner",
                    "starred_at": "2025-01-03T00:00:00Z",
                },
                {
                    "full_name": "other/gamma",
                    "description": "Gamma",
                    "html_url": "https://github.com/other/gamma",
                    "language": "Python",
                    "stargazers_count": "",
                    "topics": "testing",
                    "owner": "other",
                    "starred_at": "2025-01-01T00:00:00Z",
                },
            ]
        )

    result = GithubStarsCsvAdapter(path=str(path)).ingest(entity_types=["repository", "owner"])

    owners = [unit for unit in result.units if unit.source_entity_type == "owner"]
    repositories = [unit for unit in result.units if unit.source_entity_type == "repository"]
    assert GithubStarsCsvAdapter(path=str(path)).entity_types == ["repository", "owner"]
    assert len(owners) == 2

    owner = next(unit for unit in owners if unit.metadata["normalized_owner"] == "owner")
    owner_repositories = [unit for unit in repositories if unit.metadata["owner"].casefold() == "owner"]
    assert owner.source_id.startswith("github_stars_csv:owner:")
    assert owner.metadata["repo_count"] == 2
    assert owner.metadata["languages"] == ["Go", "Python"]
    assert owner.metadata["topics"] == ["cli", "knowledge", "server"]
    assert owner.metadata["stargazers_count"] == 25
    assert owner.metadata["first_starred_at"] == "2025-01-02T00:00:00+00:00"
    assert owner.metadata["latest_starred_at"] == "2025-01-03T00:00:00+00:00"
    assert owner.metadata["repository_source_ids"] == sorted(unit.source_id for unit in owner_repositories)
    assert {edge.to_unit_id for edge in result.edges if edge.from_unit_id == owner.source_id} == {
        unit.source_id for unit in owner_repositories
    }
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)


def test_github_stars_csv_owner_filtering(tmp_path):
    path = tmp_path / "stars.csv"
    path.write_text(
        "full_name,owner,starred_at\nowner/repo,owner,2025-01-02T00:00:00Z\n",
        encoding="utf-8",
    )

    owner_only = GithubStarsCsvAdapter(path=str(path)).ingest(entity_types=["owner"])
    repository_only = GithubStarsCsvAdapter(path=str(path)).ingest(entity_types=["repository"])

    assert [unit.source_entity_type for unit in owner_only.units] == ["owner"]
    assert owner_only.edges == []
    assert [unit.source_entity_type for unit in repository_only.units] == ["repository"]
    assert repository_only.edges == []


def test_github_stars_csv_adapter_is_registered():
    assert isinstance(get_adapter("github_stars_csv", path="/tmp/stars.csv"), GithubStarsCsvAdapter)
