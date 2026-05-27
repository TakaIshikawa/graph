from __future__ import annotations

import json

from graph.adapters.linear_comments_json import LinearCommentsJsonAdapter
from graph.adapters.registry import get_adapter


def test_linear_comments_json_ingests_comments_with_issue_and_author_metadata(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(
        json.dumps(
            {
                "comments": [
                    {
                        "id": "c1",
                        "body": "Looks good",
                        "url": "https://linear.test/comment/c1",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "updatedAt": "2026-01-02T00:00:00Z",
                        "user": {"name": "Ada"},
                        "issue": {"id": "i1", "identifier": "ENG-1", "title": "Importer"},
                        "parentId": "p1",
                        "threadId": "t1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = LinearCommentsJsonAdapter(str(path)).ingest().units[0]

    assert unit.title == "Linear comment on ENG-1"
    assert unit.content == "Looks good"
    assert unit.metadata["user"] == "Ada"
    assert unit.metadata["issue_id"] == "i1"
    assert unit.metadata["issue_identifier"] == "ENG-1"
    assert unit.metadata["issue_title"] == "Importer"
    assert unit.metadata["parent_id"] == "p1"
    assert unit.metadata["thread_id"] == "t1"
    assert isinstance(get_adapter("linear_comments_json"), LinearCommentsJsonAdapter)


def test_linear_comments_json_represents_deleted_comments_consistently(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(json.dumps([{"id": "c2", "deletedAt": "2026-01-01T00:00:00Z"}]), encoding="utf-8")

    assert LinearCommentsJsonAdapter(str(path)).ingest().units[0].content == "[deleted comment]"
