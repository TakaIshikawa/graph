from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.github_notifications_json import GithubNotificationsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_github_notifications_json_ingests_subject_metadata_and_registry(tmp_path):
    export = tmp_path / "notifications.json"
    export.write_text(
        json.dumps(
            {
                "notifications": [
                    {
                        "id": "n1",
                        "repository": {"full_name": "owner/repo"},
                        "subject": {"title": "Fix bug", "type": "PullRequest", "url": "https://api.github.com/repos/owner/repo/pulls/1"},
                        "reason": "mention",
                        "unread": True,
                        "url": "https://api.github.com/notifications/threads/n1",
                        "updated_at": "2025-01-02T03:04:05Z",
                        "last_read_at": "2025-01-03T00:00:00Z",
                        "subscription_url": "https://api.github.com/notifications/threads/n1/subscription",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GithubNotificationsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GITHUB_NOTIFICATIONS_JSON
    assert unit.source_entity_type == "notification"
    assert unit.metadata["repository"] == "owner/repo"
    assert unit.metadata["subject_title"] == "Fix bug"
    assert unit.metadata["subject_type"] == "PullRequest"
    assert unit.metadata["reason"] == "mention"
    assert unit.metadata["unread"] is True
    assert unit.metadata["url"] == "https://api.github.com/notifications/threads/n1"
    assert unit.metadata["source_file"] == "notifications.json"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("github_notifications_json", path=str(export)).name == "github_notifications_json"


def test_github_notifications_json_skips_bad_records_files_since_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"id": "old", "subject": {"title": "Old"}, "updated_at": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"items": [{"subject_title": "New", "url": "https://example.com/new", "updated_at": "2025-01-03T00:00:00Z"}, {"reason": "subscribed"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = GithubNotificationsJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="github_notifications_json", source_entity_type="notification", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["issue"]).units == []
