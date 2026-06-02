from __future__ import annotations

from datetime import datetime, timezone
import json

from graph.adapters.github_releases_json import GithubReleasesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_github_releases_json_ingests_release_metadata(tmp_path):
    path = tmp_path / "releases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 101,
                    "tag_name": "v1.2.0",
                    "name": "Version 1.2",
                    "body": "Release notes",
                    "draft": False,
                    "prerelease": True,
                    "author": {"login": "ada"},
                    "html_url": "https://github.com/acme/graph/releases/tag/v1.2.0",
                    "repository": {"full_name": "acme/graph"},
                    "assets": [{"name": "graph.tar.gz", "browser_download_url": "https://example.test/graph.tar.gz", "size": 12}],
                    "created_at": "2025-01-01T00:00:00Z",
                    "published_at": "2025-01-02T03:04:05Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = GithubReleasesJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.GITHUB_RELEASES_JSON
    assert unit.source_id == "github_releases_json:acme/graph@v1.2.0"
    assert unit.source_entity_type == "release"
    assert unit.title == "Version 1.2"
    assert "Release notes" in unit.content
    assert unit.metadata["draft"] is False
    assert unit.metadata["prerelease"] is True
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["asset_names"] == ["graph.tar.gz"]
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_github_releases_json_since_filtering_entity_filter_and_registry_aliases(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(
        json.dumps(
            {
                "releases": [
                    {"tag_name": "old", "published_at": "2025-01-01T00:00:00Z"},
                    {"tag_name": "new", "created_at": "2025-01-03T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="github_releases_json",
        source_entity_type="release",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    filtered = GithubReleasesJsonAdapter(path=str(path)).ingest(since=since)

    assert [unit.metadata["tag_name"] for unit in filtered.units] == ["new"]
    assert GithubReleasesJsonAdapter(path=str(path)).ingest(entity_types=["issue"]).units == []
    assert type(get_adapter("github-releases-json")).__name__ == "GithubReleasesJsonAdapter"
    assert type(get_adapter("github_releases_json")).__name__ == "GithubReleasesJsonAdapter"
