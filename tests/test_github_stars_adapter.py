from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.github_stars_csv import GithubStarsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


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


def test_github_stars_csv_adapter_is_registered():
    assert isinstance(get_adapter("github_stars_csv", path="/tmp/stars.csv"), GithubStarsCsvAdapter)
