from __future__ import annotations

import json

from graph.adapters.github_review_comments_json import GithubReviewCommentsJsonAdapter
from graph.adapters.registry import get_adapter


def test_github_review_comments_json_ingests_comments(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(json.dumps({"comments": [{"id": 10, "repository": {"full_name": "acme/repo"}, "pull_request_number": 7, "path": "src/app.py", "line": 12, "position": 4, "user": {"login": "ada"}, "body": "Please adjust", "html_url": "https://github.test/c", "created_at": "2026-05-01T00:00:00Z", "resolved": False}]}), encoding="utf-8")

    unit = GithubReviewCommentsJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "github_review_comments_json"
    assert unit.source_id == "github_review_comments_json:10"
    assert unit.source_entity_type == "review_comment"
    assert unit.metadata["repository"] == "acme/repo"
    assert unit.metadata["pull_request_number"] == 7
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["resolved"] is False
    assert isinstance(get_adapter("github_review_comments_json"), GithubReviewCommentsJsonAdapter)
