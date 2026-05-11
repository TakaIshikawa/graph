from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.bluesky_archive import BlueskyArchiveAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def test_bluesky_archive_imports_posts_likes_and_reposts_with_edges(tmp_path):
    export = tmp_path / "records.json"
    post_uri = "at://did:plc:alice/app.bsky.feed.post/3kpost"
    export.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "uri": post_uri,
                        "cid": "bafy-post",
                        "value": {
                            "$type": "app.bsky.feed.post",
                            "text": "A useful Bluesky note",
                            "createdAt": "2025-01-01T00:00:00Z",
                        },
                    },
                    {
                        "uri": "at://did:plc:alice/app.bsky.feed.like/3klike",
                        "cid": "bafy-like",
                        "value": {
                            "$type": "app.bsky.feed.like",
                            "createdAt": "2025-01-02T00:00:00Z",
                            "subject": {"uri": post_uri, "cid": "bafy-post"},
                        },
                    },
                    {
                        "uri": "at://did:plc:alice/app.bsky.feed.repost/3krepost",
                        "cid": "bafy-repost",
                        "value": {
                            "$type": "app.bsky.feed.repost",
                            "createdAt": "2025-01-03T00:00:00Z",
                            "subject": {"uri": post_uri, "cid": "bafy-post"},
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = BlueskyArchiveAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["post", "like", "repost"]
    post, like, repost = result.units
    assert post.source_project == SourceProject.BLUESKY_ARCHIVE
    assert post.content == "A useful Bluesky note"
    assert post.metadata["created_at"] == "2025-01-01T00:00:00Z"
    assert post.metadata["author_did"] == "did:plc:alice"
    assert post.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert BlueskyArchiveAdapter(path=str(export)).ingest().units[0].source_id == post.source_id

    assert like.metadata["subject_uri"] == post_uri
    assert like.metadata["subject_cid"] == "bafy-post"
    assert repost.metadata["subject_uri"] == post_uri
    assert len(result.edges) == 2
    assert {edge.from_unit_id for edge in result.edges} == {like.source_id, repost.source_id}
    assert {edge.to_unit_id for edge in result.edges} == {post.source_id}
    assert {edge.relation for edge in result.edges} == {EdgeRelation.REFERENCES}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}


def test_bluesky_archive_reads_directory_and_handles_missing_optional_fields(tmp_path):
    records_dir = tmp_path / "app.bsky.feed.post"
    records_dir.mkdir()
    (records_dir / "3kpost.json").write_text(
        json.dumps({"$type": "app.bsky.feed.post", "text": "Post without wrapper"}),
        encoding="utf-8",
    )
    (tmp_path / "like.json").write_text(
        json.dumps(
            {
                "$type": "app.bsky.feed.like",
                "subject": {"uri": "at://did:plc:missing/app.bsky.feed.post/3kpost"},
            }
        ),
        encoding="utf-8",
    )

    result = BlueskyArchiveAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["post", "like"]
    assert result.units[0].content == "Post without wrapper"
    assert result.edges == []


def test_bluesky_archive_filters_and_registry(tmp_path):
    export = tmp_path / "records.json"
    export.write_text(
        json.dumps([{"$type": "app.bsky.feed.post", "text": "Only post", "createdAt": "2025-01-01T00:00:00Z"}]),
        encoding="utf-8",
    )

    assert BlueskyArchiveAdapter(path=str(export)).ingest(entity_types=["like"]).units == []
    assert get_adapter("bluesky_archive", path=str(export)).name == "bluesky_archive"
