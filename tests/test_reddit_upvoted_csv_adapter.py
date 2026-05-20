from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.reddit_upvoted_csv import RedditUpvotedCsvAdapter
from graph.types.models import SyncState


def test_reddit_upvoted_csv_ingests_aliases_and_metadata(tmp_path):
    export = tmp_path / "upvoted.csv"
    export.write_text(
        "thing_id,post_title,comment_body,subreddit_name_prefixed,username,link_url,link_permalink,ups,created_utc,upvoted_at\n"
        "t1_comment,Post title,Useful comment,r/python,ada,https://example.com,/r/python/comments/abc/_/comment/,12,1735689600,2025-01-03T00:00:00Z\n",
        encoding="utf-8",
    )

    unit = RedditUpvotedCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "reddit_upvoted_csv"
    assert unit.source_entity_type == "reddit_upvote"
    assert unit.title == "Post title"
    assert "Useful comment" in unit.content
    assert unit.metadata["subreddit"] == "r/python"
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["url"] == "https://example.com"
    assert unit.metadata["permalink"] == "https://www.reddit.com/r/python/comments/abc/_/comment/"
    assert unit.metadata["score"] == 12
    assert unit.metadata["created_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["upvoted_at"] == "2025-01-03T00:00:00+00:00"
    assert unit.metadata["source_file"] == "upvoted.csv"
    assert unit.metadata["row_index"] == 1
    assert unit.metadata["raw_record"]["thing_id"] == "t1_comment"
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 3, tzinfo=timezone.utc)


def test_reddit_upvoted_csv_directory_skips_bad_files_dedupes_and_filters(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "nested"
    second.mkdir()
    (second / "b.csv").write_text(
        "id,title,subreddit,created_at\n"
        "abc,Duplicate One,python,2025-01-01\n"
        "abc,Duplicate Two,python,2025-01-02\n"
        "def,New One,learnpython,2025-01-05\n",
        encoding="utf-8",
    )
    first.write_bytes(b"\xff\xfe\x00")

    adapter = RedditUpvotedCsvAdapter(path=str(tmp_path))
    since = SyncState(source_project="reddit_upvoted_csv", source_entity_type="reddit_upvote", last_sync_at=datetime(2025, 1, 3, tzinfo=timezone.utc))
    result = adapter.ingest(since=since)

    assert [unit.title for unit in result.units] == ["New One"]
    assert adapter.ingest(entity_types=["post"]).units == []


def test_reddit_upvoted_csv_fallback_digest_is_stable_for_body_only_rows(tmp_path):
    export = tmp_path / "comments.csv"
    export.write_text("text,date\nBody only,01/02/2025\n", encoding="utf-8")

    first = RedditUpvotedCsvAdapter(path=str(export)).ingest().units[0]
    second = RedditUpvotedCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("reddit_upvoted_csv:")
