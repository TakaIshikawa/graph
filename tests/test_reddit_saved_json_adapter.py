from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.reddit_saved_json import RedditSavedJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


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


def test_reddit_saved_json_emits_redditor_aggregates_and_edges(tmp_path):
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
                        "subreddit": "python",
                        "author": "ada",
                        "created_utc": 1735689600,
                    },
                },
                {
                    "id": "def",
                    "name": "t1_def",
                    "body": "Saved comment body",
                    "link_title": "Discussion title",
                    "subreddit": "learnpython",
                    "author": "Ada",
                    "created_utc": 1735776000,
                },
                {
                    "id": "ghi",
                    "name": "t3_ghi",
                    "title": "Other post",
                    "subreddit": "rust",
                    "author": "grace",
                    "created_utc": 1735862400,
                },
            ]
        ),
        encoding="utf-8",
    )

    first = RedditSavedJsonAdapter(path=str(export)).ingest(entity_types=["post", "comment", "redditor"])
    second = RedditSavedJsonAdapter(path=str(export)).ingest(entity_types=["post", "comment", "redditor"])

    assert "redditor" in RedditSavedJsonAdapter(path=str(export)).entity_types
    redditors = {unit.metadata["normalized_author"]: unit for unit in first.units if unit.source_entity_type == "redditor"}
    assert set(redditors) == {"ada", "grace"}
    ada = redditors["ada"]
    assert ada.metadata["saved_count"] == 2
    assert ada.metadata["post_count"] == 1
    assert ada.metadata["comment_count"] == 1
    assert ada.metadata["subreddits"] == ["learnpython", "python"]
    assert ada.metadata["first_saved_at"] == "2025-01-01T00:00:00+00:00"
    assert ada.metadata["last_saved_at"] == "2025-01-02T00:00:00+00:00"
    assert len(ada.metadata["item_source_ids"]) == 2

    edges = [edge for edge in first.edges if edge.metadata["relation_type"] == "saved_item_author"]
    assert len(edges) == 3
    assert {edge.relation for edge in edges} == {EdgeRelation.RELATES_TO}
    assert {edge.to_unit_id for edge in edges} == {unit.source_id for unit in redditors.values()}
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


def test_reddit_saved_json_redditor_entity_filtering(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(json.dumps([{"id": "abc", "name": "t3_abc", "title": "Post", "author": "ada"}]), encoding="utf-8")

    result = RedditSavedJsonAdapter(path=str(export)).ingest(entity_types=["redditor"])

    assert [unit.source_entity_type for unit in result.units] == ["redditor"]
    assert result.edges == []


def test_reddit_saved_json_handles_deleted_author_and_crosspost_source(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "abc",
                    "name": "t3_abc",
                    "title": "Crosspost",
                    "author": "[deleted]",
                    "subreddit": "python",
                    "crosspost_parent": "t3_parent",
                    "crosspost_parent_list": [{"url": "https://example.com/source"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = RedditSavedJsonAdapter(path=str(export)).ingest(entity_types=["post", "redditor"])

    post = next(unit for unit in result.units if unit.source_entity_type == "post")
    redditor = next(unit for unit in result.units if unit.source_entity_type == "redditor")
    assert post.metadata["author"] == "[deleted]"
    assert post.metadata["crosspost_parent"] == "t3_parent"
    assert post.metadata["crosspost_source_url"] == "https://example.com/source"
    assert redditor.metadata["normalized_author"] == "[deleted]"
