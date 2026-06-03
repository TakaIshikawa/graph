from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_gists_json import GithubGistsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_github_gists_json_ingests_array_metadata_content_and_registry(tmp_path):
    export = tmp_path / "gists.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "abc",
                    "description": "Useful gist",
                    "files": {
                        "main.py": {"filename": "main.py", "language": "Python", "size": 12, "content": "print('hi')"},
                        "README.md": {"filename": "README.md", "language": "Markdown"},
                    },
                    "public": True,
                    "owner": {"login": "ada"},
                    "comments": 3,
                    "html_url": "https://gist.github.com/abc",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GithubGistsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITHUB_GISTS_JSON
    assert unit.source_entity_type == "gist"
    assert unit.metadata["id"] == "abc"
    assert unit.metadata["file_names"] == ["main.py", "README.md"]
    assert unit.metadata["languages"] == ["Markdown", "Python"]
    assert unit.metadata["public"] is True
    assert unit.metadata["owner"] == "ada"
    assert unit.metadata["comments_count"] == 3
    assert unit.metadata["html_url"] == "https://gist.github.com/abc"
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "print('hi')" in unit.content
    assert get_adapter("github_gists_json", path=str(export)).name == "github_gists_json"


def test_github_gists_json_wrappers_directory_since_and_filters(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps({"gists": [{"id": "old", "description": "Old", "updated_at": "2025-01-01T00:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "two.json").write_text(json.dumps({"items": [{"id": "new", "description": "New", "files": [{"name": "a.js", "language": "JavaScript"}], "updated_at": "2025-01-03T00:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = GithubGistsJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="github_gists_json", source_entity_type="gist", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].metadata["languages"] == ["JavaScript"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["file"]).units == []


def test_github_gists_json_handles_private_empty_description_and_truncated_content(tmp_path):
    export = tmp_path / "gists.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "secret",
                    "description": "",
                    "public": False,
                    "files": {"notes.md": {"filename": "notes.md", "truncated_content": "partial note"}},
                    "updated_at": "2025-01-01T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = GithubGistsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "secret"
    assert unit.metadata["public"] is False
    assert "description" not in unit.metadata
    assert unit.metadata["file_names"] == ["notes.md"]
    assert "partial note" in unit.content
