from __future__ import annotations

import csv

from graph.adapters.github_stars_csv import GithubStarsCsvAdapter
from graph.types.enums import SourceProject


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_github_stars_csv_imports_repository_metadata(tmp_path):
    path = tmp_path / "stars.csv"
    _write_csv(
        path,
        [
            {
                "full_name": "octocat/Hello-World",
                "html_url": "https://github.com/octocat/Hello-World",
                "description": "Example repository",
                "language": "Ruby",
                "topics": "example; github | testing",
                "stars": "42",
                "archived": "false",
                "private": "0",
                "starred_at": "2025-01-02T03:04:05Z",
            }
        ],
    )

    unit = GithubStarsCsvAdapter(path=str(path)).ingest(entity_types=["repository"]).units[0]

    assert unit.source_project == SourceProject.GITHUB_STARS_CSV
    assert unit.source_entity_type == "repository"
    assert unit.title == "octocat/Hello-World"
    assert unit.metadata["owner"] == "octocat"
    assert unit.metadata["repo"] == "Hello-World"
    assert unit.metadata["description"] == "Example repository"
    assert unit.metadata["language"] == "Ruby"
    assert unit.metadata["topics"] == ["example", "github", "testing"]
    assert unit.tags == ["example", "github", "testing"]
    assert unit.metadata["stars"] == 42
    assert unit.metadata["stargazers_count"] == 42
    assert unit.metadata["archived"] is False
    assert unit.metadata["private"] is False
    assert unit.metadata["starred_at"] == "2025-01-02T03:04:05Z"
    assert unit.metadata["url"] == "https://github.com/octocat/Hello-World"
    assert unit.metadata["source_file"] == "stars.csv"


def test_github_stars_csv_uses_stable_source_ids_for_url_only_rows(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    row = {
        "url": "https://github.com/psf/requests",
        "description": "Python HTTP library",
        "topics": "python, http, python",
    }
    _write_csv(first, [row])
    _write_csv(second, [row])

    first_unit = GithubStarsCsvAdapter(path=str(first)).ingest(entity_types=["repository"]).units[0]
    second_unit = GithubStarsCsvAdapter(path=str(second)).ingest(entity_types=["repository"]).units[0]

    assert first_unit.title == "psf/requests"
    assert first_unit.metadata["owner"] == "psf"
    assert first_unit.metadata["repo"] == "requests"
    assert first_unit.metadata["topics"] == ["python", "http"]
    assert first_unit.source_id == second_unit.source_id
