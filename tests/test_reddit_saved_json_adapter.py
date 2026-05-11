from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.reddit_saved_json import RedditSavedJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_reddit_saved_json_imports_posts_and_comments(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(
        json.dumps(
            [
                {
                    "kind": "t3",
                    "data": {
                        "id": "abc",
                        "name": "t3_abc",
                        "title": "Useful post",
                        "selftext": "Post body",
                        "subreddit": "python",
                        "author": "ada",
                        "permalink": "/r/python/comments/abc/useful_post/",
                        "url": "https://example.com/post",
                        "created_utc": 1735689600,
                        "score": 42,
                    },
                },
                {
                    "id": "def",
                    "name": "t1_def",
                    "body": "Saved comment body",
                    "link_title": "Discussion title",
                    "subreddit": "learnpython",
                    "author": "grace",
                    "permalink": "https://www.reddit.com/r/learnpython/comments/abc/_/def/",
                    "created_utc": 1735776000,
                    "score": "7",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = RedditSavedJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    post, comment = result.units
    assert post.source_project == SourceProject.REDDIT_SAVED_JSON
    assert post.source_entity_type == "post"
    assert post.metadata["permalink"] == "https://www.reddit.com/r/python/comments/abc/useful_post/"
    assert post.metadata["url"] == "https://example.com/post"
    assert post.metadata["created_utc"] == "2025-01-01T00:00:00+00:00"
    assert post.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert comment.source_entity_type == "comment"
    assert comment.title == "Discussion title"
    assert comment.metadata["body"] == "Saved comment body"
    assert comment.metadata["score"] == 7


def test_reddit_saved_json_filters_and_registry(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(json.dumps({"saved": [{"id": "abc", "name": "t3_abc", "title": "Post"}]}), encoding="utf-8")

    assert RedditSavedJsonAdapter(path=str(export)).ingest(entity_types=["comment"]).units == []
    assert get_adapter("reddit_saved_json", path=str(export)).name == "reddit_saved_json"
