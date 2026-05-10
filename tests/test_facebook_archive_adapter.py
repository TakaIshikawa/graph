from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.facebook_archive import FacebookArchiveAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_facebook_archive_adapter_ingests_posts_with_metadata_and_tags(tmp_path):
    posts_dir = tmp_path / "posts"
    posts_dir.mkdir()
    posts_file = posts_dir / "your_posts_1.json"
    posts_file.write_text(
        json.dumps(
            [
                {
                    "timestamp": 1712001600,
                    "post": "First post with @[123:Alice Bob] and #project",
                    "title": "Project Update",
                    "tags": [{"id": "123", "name": "Alice Bob"}],
                    "attachments": [
                        {
                            "data": [
                                {
                                    "media": "photo.jpg",
                                    "name": "Photo 1",
                                }
                            ]
                        }
                    ],
                    "place": {
                        "name": "San Francisco",
                        "address": "123 Main St",
                    },
                    "reactions": [
                        {"reaction": "like"},
                        {"reaction": "like"},
                        {"reaction": "love"},
                    ],
                    "comments": [{"comment": "Great!"}],
                },
                {
                    "timestamp": 1712088000,
                    "post": "Second post",
                    "post_id": "post_456",
                },
                {
                    "timestamp": 1712174400,
                    "post": "   ",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = FacebookArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert [unit.source_entity_type for unit in result.units] == ["post", "post"]

    first = result.units[0]
    assert first.source_project == SourceProject.FACEBOOK_ARCHIVE
    assert first.source_id.startswith("facebook_archive:post:")
    assert first.title == "Project Update"
    assert "First post with @[123:Alice Bob] and #project" in first.content
    assert first.created_at == datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    assert first.updated_at == datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    assert "facebook" in first.tags
    assert "facebook-post" in first.tags
    assert "tag-alice-bob" in first.tags
    assert first.metadata["attachments"][0]["media"] == "photo.jpg"
    assert first.metadata["place"]["name"] == "San Francisco"
    assert first.metadata["reactions"]["like"] == 2
    assert first.metadata["reactions"]["love"] == 1
    assert first.metadata["comments_count"] == 1
    assert first.metadata["source_path"] == "posts/your_posts_1.json"

    second = result.units[1]
    assert second.source_id == "facebook_archive:post:post_456"
    assert second.content == "Second post"

    # Check edges for tagged people
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == first.source_id
    assert edge.to_unit_id == "facebook_person:123"
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "facebook_tag"
    assert edge.metadata["tagged_person_name"] == "Alice Bob"


def test_facebook_archive_adapter_ingests_messages_with_conversation_edges(tmp_path):
    messages_dir = tmp_path / "messages" / "inbox" / "alice_123"
    messages_dir.mkdir(parents=True)
    message_file = messages_dir / "message_1.json"
    message_file.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "sender_name": "Alice",
                        "timestamp_ms": 1712001600000,
                        "content": "Hello!",
                        "photos": [{"uri": "photo1.jpg"}],
                        "reactions": [{"actor": "Bob", "reaction": "👍"}],
                    },
                    {
                        "sender_name": "Bob",
                        "timestamp_ms": 1712001700000,
                        "content": "Hi there!",
                        "videos": [{"uri": "video1.mp4"}],
                    },
                    {
                        "sender_name": "Alice",
                        "timestamp_ms": 1712001800000,
                        "content": "How are you?",
                        "share": "https://example.com",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = FacebookArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 3
    assert all(unit.source_entity_type == "message" for unit in result.units)
    assert all(unit.source_project == SourceProject.FACEBOOK_ARCHIVE for unit in result.units)

    first = result.units[0]
    assert "Alice: Hello!" in first.title
    assert "From: Alice" in first.content
    assert "Hello!" in first.content
    assert first.metadata["sender_name"] == "Alice"
    assert first.metadata["conversation_id"] == "alice_123"
    assert first.metadata["photos"] == ["photo1.jpg"]
    assert first.metadata["reactions"][0]["actor"] == "Bob"
    assert first.metadata["reactions"][0]["reaction"] == "👍"
    assert first.created_at == datetime(2024, 4, 1, 20, 0, 0, tzinfo=timezone.utc)
    assert "facebook-message" in first.tags

    second = result.units[1]
    assert second.metadata["sender_name"] == "Bob"
    assert second.metadata["videos"] == ["video1.mp4"]

    third = result.units[2]
    assert third.metadata["share"] == "https://example.com"

    # Check conversation edges
    assert len(result.edges) == 2
    # Edges should connect messages in the same conversation
    all_message_ids = {first.source_id, second.source_id, third.source_id}
    for edge in result.edges:
        assert edge.from_unit_id in all_message_ids
        assert edge.to_unit_id in all_message_ids
        assert edge.from_unit_id != edge.to_unit_id
        assert edge.metadata["relation_type"] == "facebook_conversation"
        assert edge.metadata["conversation_id"] == "alice_123"
        assert edge.relation == EdgeRelation.REFERENCES
        assert edge.source == EdgeSource.SOURCE


def test_facebook_archive_adapter_filters_by_entity_type_and_date(tmp_path):
    posts_dir = tmp_path / "posts"
    posts_dir.mkdir()
    (posts_dir / "your_posts_1.json").write_text(
        json.dumps(
            [
                {"timestamp": 1712001600, "post": "Old post"},
                {"timestamp": 1712260800, "post": "New post"},
            ]
        ),
        encoding="utf-8",
    )

    messages_dir = tmp_path / "messages" / "inbox" / "conv1"
    messages_dir.mkdir(parents=True)
    (messages_dir / "message_1.json").write_text(
        json.dumps({"messages": [{"sender_name": "Alice", "timestamp_ms": 1712001600000, "content": "Hi"}]}),
        encoding="utf-8",
    )

    adapter = FacebookArchiveAdapter(path=str(tmp_path))

    # Test entity type filtering
    posts_only = adapter.ingest(entity_types=["post"])
    assert len(posts_only.units) == 2
    assert all(u.source_entity_type == "post" for u in posts_only.units)

    messages_only = adapter.ingest(entity_types=["message"])
    assert len(messages_only.units) == 1
    assert all(u.source_entity_type == "message" for u in messages_only.units)

    # Test date filtering
    filtered = adapter.ingest(
        since=SyncState(
            source_project="facebook_archive",
            source_entity_type="post",
            last_sync_at=datetime(2024, 4, 2, tzinfo=timezone.utc),
        )
    )
    assert len(filtered.units) == 1  # Only new post (4/5) is included


def test_facebook_archive_adapter_handles_multiple_post_files(tmp_path):
    posts_dir = tmp_path / "posts"
    posts_dir.mkdir()
    (posts_dir / "your_posts_1.json").write_text(
        json.dumps([{"timestamp": 1712001600, "post": "Post 1"}]),
        encoding="utf-8",
    )
    (posts_dir / "your_posts_2.json").write_text(
        json.dumps([{"timestamp": 1712088000, "post": "Post 2"}]),
        encoding="utf-8",
    )

    result = FacebookArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert {u.content for u in result.units} == {"Post 1", "Post 2"}


def test_facebook_archive_adapter_handles_attachments_and_reactions(tmp_path):
    posts_dir = tmp_path / "posts"
    posts_dir.mkdir()
    (posts_dir / "your_posts_1.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": 1712001600,
                    "post": "Post with media",
                    "attachments": [
                        {
                            "data": [
                                {
                                    "media": "https://example.com/photo.jpg",
                                    "name": "Vacation Photo",
                                },
                                {
                                    "external_context": {
                                        "url": "https://example.com/article"
                                    },
                                    "name": "Interesting Article",
                                },
                            ]
                        }
                    ],
                    "reactions": [
                        {"reaction": "like"},
                        {"reaction": "wow"},
                        {"reaction": "like"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = FacebookArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert len(unit.metadata["attachments"]) == 2
    assert unit.metadata["attachments"][0]["media"] == "https://example.com/photo.jpg"
    assert unit.metadata["attachments"][0]["name"] == "Vacation Photo"
    assert unit.metadata["attachments"][1]["url"] == "https://example.com/article"
    assert unit.metadata["reactions"]["like"] == 2
    assert unit.metadata["reactions"]["wow"] == 1


def test_facebook_archive_adapter_empty_and_invalid_files(tmp_path):
    posts_dir = tmp_path / "posts"
    posts_dir.mkdir()
    (posts_dir / "empty.json").write_text("[]", encoding="utf-8")
    (posts_dir / "invalid.json").write_text("not json", encoding="utf-8")

    result = FacebookArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_facebook_archive_adapter_nonexistent_path():
    result = FacebookArchiveAdapter(path="/nonexistent/path").ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0
