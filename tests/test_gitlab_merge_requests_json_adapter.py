from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.gitlab_merge_requests_json import GitlabMergeRequestsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_gitlab_merge_requests_json_ingests_nested_metadata_and_registry(tmp_path):
    export = tmp_path / "merge-requests.json"
    export.write_text(
        json.dumps(
            {
                "merge_requests": [
                    {
                        "iid": 7,
                        "path_with_namespace": "group/project",
                        "title": "Add feature",
                        "description": "Useful change",
                        "state": "merged",
                        "web_url": "https://gitlab.com/group/project/-/merge_requests/7",
                        "author": {"username": "ada"},
                        "assignees": [{"username": "grace"}],
                        "labels": ["backend", "ready"],
                        "source_branch": "feature",
                        "target_branch": "main",
                        "merged_at": "2025-01-03T00:00:00Z",
                        "created_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-02T03:04:05Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GitlabMergeRequestsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITLAB_MERGE_REQUESTS_JSON
    assert unit.source_entity_type == "merge_request"
    assert unit.metadata["project"] == "group/project"
    assert unit.metadata["title"] == "Add feature"
    assert unit.metadata["state"] == "merged"
    assert unit.metadata["source_branch"] == "feature"
    assert unit.metadata["target_branch"] == "main"
    assert unit.metadata["labels"] == ["backend", "ready"]
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["assignees"] == ["grace"]
    assert unit.metadata["web_url"] == "https://gitlab.com/group/project/-/merge_requests/7"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("gitlab_merge_requests_json", path=str(export)).name == "gitlab_merge_requests_json"


def test_gitlab_merge_requests_json_arrays_skips_bad_records_since_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"title": "Old", "web_url": "https://example.com/old", "updated_at": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"mrs": [{"title": "New", "web_url": "https://example.com/new", "updated_at": "2025-01-03T00:00:00Z"}, {"state": "opened"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = GitlabMergeRequestsJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="gitlab_merge_requests_json", source_entity_type="merge_request", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["issue"]).units == []
