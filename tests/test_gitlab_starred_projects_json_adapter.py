from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.gitlab_starred_projects_json import GitlabStarredProjectsJsonAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_gitlab_starred_projects_json_ingests_project_metadata(tmp_path):
    export = tmp_path / "stars.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": 42,
                    "path_with_namespace": "group/project",
                    "name": "project",
                    "description": "Useful library",
                    "web_url": "https://gitlab.com/group/project",
                    "star_count": 99,
                    "forks_count": 7,
                    "last_activity_at": "2025-04-05T12:30:00Z",
                    "created_at": "2024-01-02T03:04:05Z",
                    "namespace": {"full_path": "group"},
                    "topics": ["python", "cli"],
                    "default_branch": "main",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = GitlabStarredProjectsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "gitlab_starred_projects_json"
    assert unit.source_id.startswith("gitlab_starred_projects_json:")
    assert unit.source_entity_type == "repository"
    assert unit.title == "group/project"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Useful library" in unit.content
    assert "Namespace: group" in unit.content
    assert "Topics: python, cli" in unit.content
    assert "Default branch: main" in unit.content
    assert "URL: https://gitlab.com/group/project" in unit.content
    assert unit.metadata["project_id"] == "42"
    assert unit.metadata["path_with_namespace"] == "group/project"
    assert unit.metadata["namespace_path"] == "group"
    assert unit.metadata["name"] == "project"
    assert unit.metadata["description"] == "Useful library"
    assert unit.metadata["web_url"] == "https://gitlab.com/group/project"
    assert unit.metadata["star_count"] == 99
    assert unit.metadata["forks_count"] == 7
    assert unit.metadata["topics"] == ["python", "cli"]
    assert unit.metadata["default_branch"] == "main"
    assert unit.metadata["created_at"] == "2024-01-02T03:04:05+00:00"
    assert unit.metadata["last_activity_at"] == "2025-04-05T12:30:00+00:00"
    assert unit.metadata["source_file"] == "stars.json"
    assert unit.metadata["record_index"] == 0
    assert unit.tags == ["gitlab", "python", "cli"]
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 4, 5, 12, 30, tzinfo=timezone.utc)


def test_gitlab_starred_projects_json_skips_invalid_records_and_filters_since(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "projects.json").write_text(
        json.dumps(
            {
                "projects": [
                    {},
                    {
                        "id": 1,
                        "path_with_namespace": "old/project",
                        "last_activity_at": "2025-01-01T00:00:00Z",
                        "tag_list": ["old"],
                    },
                    {
                        "id": 2,
                        "path_with_namespace": "new/project",
                        "last_activity_at": "2025-01-03T00:00:00Z",
                        "tag_list": "data, ml",
                    },
                    {
                        "id": 3,
                        "path_with_namespace": "created/project",
                        "created_at": "2025-01-04T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = GitlabStarredProjectsJsonAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="gitlab_starred_projects_json",
        source_entity_type="repository",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    result = adapter.ingest(since=sync)

    assert sorted(unit.metadata["path_with_namespace"] for unit in result.units) == ["created/project", "new/project"]
    new_project = next(unit for unit in result.units if unit.metadata["path_with_namespace"] == "new/project")
    assert new_project.metadata["topics"] == ["data", "ml"]
    assert len({unit.source_id for unit in result.units}) == 2
    assert adapter.ingest(entity_types=["issue"]).units == []
