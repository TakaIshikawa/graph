from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.reddit_saved_csv import RedditSavedCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def test_reddit_saved_csv_imports_posts_comments_and_reply_edges(tmp_path):
    posts = tmp_path / "saved_posts.csv"
    comments = tmp_path / "saved_comments.csv"
    with posts.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "name", "title", "selftext", "subreddit", "author", "permalink", "url", "created_utc", "score"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "abc",
                "name": "t3_abc",
                "title": "Useful post",
                "selftext": "Post body",
                "subreddit": "python",
                "author": "ada",
                "permalink": "/r/python/comments/abc/useful_post/",
                "url": "https://example.com/post",
                "created_utc": "1735689600",
                "score": "42",
            }
        )
    with comments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "name",
                "body",
                "link_title",
                "link_id",
                "parent_id",
                "subreddit",
                "author",
                "permalink",
                "created_utc",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "def",
                "name": "t1_def",
                "body": "Saved comment body",
                "link_title": "Useful post",
                "link_id": "t3_abc",
                "parent_id": "t3_abc",
                "subreddit": "learnpython",
                "author": "grace",
                "permalink": "https://www.reddit.com/r/python/comments/abc/_/def/",
                "created_utc": "1735776000",
                "score": "7",
            }
        )

    result = RedditSavedCsvAdapter(path=str(tmp_path)).ingest(entity_types=["post", "comment"])

    assert len(result.units) == 2
    post, comment = result.units
    assert post.source_project == SourceProject.REDDIT_SAVED_CSV
    assert post.source_entity_type == "post"
    assert post.metadata["permalink"] == "https://www.reddit.com/r/python/comments/abc/useful_post/"
    assert post.metadata["url"] == "https://example.com/post"
    assert post.metadata["created_utc"] == "2025-01-01T00:00:00+00:00"
    assert post.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert comment.source_entity_type == "comment"
    assert comment.title == "Useful post"
    assert comment.metadata["body"] == "Saved comment body"
    assert comment.metadata["score"] == 7
    assert len(result.edges) == 1
    assert result.edges[0].from_unit_id == comment.source_id
    assert result.edges[0].to_unit_id == post.source_id
    assert result.edges[0].relation == EdgeRelation.REPLIES_TO
    assert result.edges[0].source == EdgeSource.SOURCE


def test_reddit_saved_csv_filters_and_registry(tmp_path):
    export = tmp_path / "saved_posts.csv"
    export.write_text("id,name,title\nabc,t3_abc,Post\n", encoding="utf-8")

    assert RedditSavedCsvAdapter(path=str(export)).ingest(entity_types=["comment"]).units == []
    assert get_adapter("reddit_saved_csv", path=str(export)).name == "reddit_saved_csv"


def test_reddit_saved_csv_ingests_subreddit_aggregates_and_edges(tmp_path):
    export = tmp_path / "saved.csv"
    with export.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "name", "type", "title", "body", "subreddit", "permalink", "created_utc"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "abc",
                "name": "t3_abc",
                "type": "post",
                "title": "Useful post",
                "subreddit": "Python",
                "permalink": "/r/Python/comments/abc/useful_post/",
                "created_utc": "1735689600",
            }
        )
        writer.writerow(
            {
                "id": "def",
                "name": "t1_def",
                "type": "comment",
                "body": "Saved comment body",
                "subreddit": "r/python",
                "permalink": "https://www.reddit.com/r/Python/comments/abc/_/def/",
                "created_utc": "1735776000",
            }
        )

    subreddits = RedditSavedCsvAdapter(path=str(export)).ingest(entity_types=["subreddit"])
    with_items = RedditSavedCsvAdapter(path=str(export)).ingest(entity_types=["post", "comment", "subreddit"])

    assert RedditSavedCsvAdapter(path=str(export)).entity_types == ["post", "comment", "subreddit"]
    subreddit = subreddits.units[0]
    assert subreddit.source_entity_type == "subreddit"
    assert subreddit.title == "r/Python"
    assert subreddit.metadata["normalized_subreddit"] == "python"
    assert subreddit.metadata["saved_count"] == 2
    assert subreddit.metadata["post_count"] == 1
    assert subreddit.metadata["comment_count"] == 1
    assert len(subreddit.metadata["saved_source_ids"]) == 2
    assert subreddit.metadata["saved_source_ids"] == sorted(subreddit.metadata["saved_source_ids"])
    assert subreddits.edges == []

    subreddit_edges = [edge for edge in with_items.edges if edge.metadata.get("relation_type") == "subreddit_saved_item"]
    assert len(subreddit_edges) == 2
    assert {edge.from_unit_id for edge in subreddit_edges} == {subreddit.source_id}
    assert {edge.relation for edge in subreddit_edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in subreddit_edges} == {EdgeSource.SOURCE}
