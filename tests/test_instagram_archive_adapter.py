from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.instagram_archive import InstagramArchiveAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_instagram_archive_adapter_ingests_posts_with_metadata(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    posts_file = content_dir / "posts_1.json"
    posts_file.write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Beautiful sunset at the beach #sunset #beach @friend123",
                    "media": [
                        {
                            "uri": "photos/sunset.jpg",
                            "media_type": "photo",
                        }
                    ],
                    "location": {
                        "name": "Santa Monica Beach",
                        "latitude": 34.0195,
                        "longitude": -118.4912,
                    },
                },
                {
                    "creation_timestamp": 1712088000,
                    "caption": "Coffee time ☕",
                    "media": [
                        {
                            "uri": "photos/coffee.jpg",
                            "media_type": "photo",
                        }
                    ],
                    "id": "post_456",
                },
                {
                    "creation_timestamp": 1712174400,
                    "title": "   ",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert [unit.source_entity_type for unit in result.units] == ["post", "post"]

    first = result.units[0]
    assert first.source_project == SourceProject.INSTAGRAM_ARCHIVE
    assert first.source_id.startswith("instagram_archive:post:")
    assert first.title == "Beautiful sunset at the beach #sunset #beach @friend123"
    assert "Beautiful sunset at the beach #sunset #beach @friend123" in first.content
    assert first.created_at == datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    assert first.updated_at == datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    assert "instagram" in first.tags
    assert "instagram-post" in first.tags
    assert "hashtag-sunset" in first.tags
    assert "hashtag-beach" in first.tags
    assert first.metadata["media"][0]["media_url"] == "photos/sunset.jpg"
    assert first.metadata["media"][0]["media_type"] == "photo"
    assert first.metadata["location"]["name"] == "Santa Monica Beach"
    assert first.metadata["hashtags"] == ["beach", "sunset"]
    assert first.metadata["mentions"] == ["friend123"]
    assert first.metadata["source_path"] == "content/posts_1.json"

    second = result.units[1]
    assert second.source_id == "instagram_archive:post:post_456"
    assert "Coffee time" in second.content


def test_instagram_archive_adapter_ingests_stories_with_expiration(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    stories_file = content_dir / "stories.json"
    stories_file.write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "expiration_timestamp": 1712088000,
                    "media": [
                        {
                            "uri": "stories/story1.jpg",
                            "media_type": "photo",
                        }
                    ],
                    "id": "story_123",
                },
                {
                    "creation_timestamp": 1712174400,
                    "media": [
                        {
                            "uri": "stories/story2.mp4",
                            "media_type": "video",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert all(unit.source_entity_type == "story" for unit in result.units)
    assert all(unit.source_project == SourceProject.INSTAGRAM_ARCHIVE for unit in result.units)

    first = result.units[0]
    assert first.source_id == "instagram_archive:story:story_123"
    assert "Instagram story 2024-04-01" in first.title
    assert "Created: 2024-04-01T20:00:00+00:00" in first.content
    assert "Expires: 2024-04-02T20:00:00+00:00" in first.content
    assert first.metadata["media"][0]["media_url"] == "stories/story1.jpg"
    assert first.metadata["media"][0]["media_type"] == "photo"
    assert first.metadata["expiration_timestamp"] == 1712088000
    assert "instagram-story" in first.tags

    second = result.units[1]
    assert second.metadata["media"][0]["media_type"] == "video"


def test_instagram_archive_adapter_ingests_messages_with_conversation_edges(tmp_path):
    messages_dir = tmp_path / "messages" / "inbox" / "friend_abc"
    messages_dir.mkdir(parents=True)
    message_file = messages_dir / "message_1.json"
    message_file.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "sender_name": "Alice",
                        "timestamp_ms": 1712001600000,
                        "content": "Hey! How are you?",
                    },
                    {
                        "sender_name": "Bob",
                        "timestamp_ms": 1712001700000,
                        "content": "Good! You?",
                        "media_url": "messages/photo.jpg",
                        "media_type": "photo",
                    },
                    {
                        "sender_name": "Alice",
                        "timestamp_ms": 1712001800000,
                        "content": "Great!",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 3
    assert all(unit.source_entity_type == "message" for unit in result.units)
    assert all(unit.source_project == SourceProject.INSTAGRAM_ARCHIVE for unit in result.units)

    first = result.units[0]
    assert "Alice: Hey! How are you?" in first.title
    assert "From: Alice" in first.content
    assert "Hey! How are you?" in first.content
    assert first.metadata["sender_name"] == "Alice"
    assert first.metadata["conversation_id"] == "friend_abc"
    assert first.created_at == datetime(2024, 4, 1, 20, 0, 0, tzinfo=timezone.utc)
    assert "instagram-message" in first.tags

    second = result.units[1]
    assert second.metadata["sender_name"] == "Bob"
    assert second.metadata["media_url"] == "messages/photo.jpg"
    assert second.metadata["media_type"] == "photo"

    third = result.units[2]
    assert third.metadata["sender_name"] == "Alice"

    # Check conversation edges
    assert len(result.edges) == 2
    all_message_ids = {first.source_id, second.source_id, third.source_id}
    for edge in result.edges:
        assert edge.from_unit_id in all_message_ids
        assert edge.to_unit_id in all_message_ids
        assert edge.from_unit_id != edge.to_unit_id
        assert edge.metadata["relation_type"] == "instagram_conversation"
        assert edge.metadata["conversation_id"] == "friend_abc"
        assert edge.relation == EdgeRelation.REFERENCES
        assert edge.source == EdgeSource.SOURCE


def test_instagram_archive_adapter_creates_hashtag_edges(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    posts_file = content_dir / "posts_1.json"
    posts_file.write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Love this #travel #adventure #photography",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 3  # 3 hashtag edges

    post = result.units[0]
    hashtag_edges = [e for e in result.edges if e.metadata["relation_type"] == "instagram_hashtag"]
    assert len(hashtag_edges) == 3

    hashtags = {e.metadata["hashtag"] for e in hashtag_edges}
    assert hashtags == {"travel", "adventure", "photography"}

    for edge in hashtag_edges:
        assert edge.from_unit_id == post.source_id
        assert edge.to_unit_id.startswith("instagram_hashtag:")
        assert edge.relation == EdgeRelation.REFERENCES
        assert edge.source == EdgeSource.SOURCE


def test_instagram_archive_adapter_creates_mention_edges(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    posts_file = content_dir / "posts_1.json"
    posts_file.write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Thanks @alice @bob.smith for the amazing day!",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 2  # 2 mention edges

    post = result.units[0]
    mention_edges = [e for e in result.edges if e.metadata["relation_type"] == "instagram_mention"]
    assert len(mention_edges) == 2

    mentions = {e.metadata["username"] for e in mention_edges}
    assert mentions == {"alice", "bob.smith"}

    for edge in mention_edges:
        assert edge.from_unit_id == post.source_id
        assert edge.to_unit_id.startswith("instagram_user:")
        assert edge.relation == EdgeRelation.REFERENCES
        assert edge.source == EdgeSource.SOURCE


def test_instagram_archive_adapter_creates_location_edges(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    posts_file = content_dir / "posts_1.json"
    posts_file.write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Great view!",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                    "location": {
                        "name": "Eiffel Tower",
                        "latitude": 48.8584,
                        "longitude": 2.2945,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 1  # 1 location edge

    post = result.units[0]
    location_edge = result.edges[0]
    assert location_edge.from_unit_id == post.source_id
    assert location_edge.to_unit_id.startswith("instagram_location:")
    assert location_edge.metadata["relation_type"] == "instagram_location"
    assert location_edge.metadata["location_name"] == "Eiffel Tower"
    assert location_edge.metadata["latitude"] == "48.8584"
    assert location_edge.metadata["longitude"] == "2.2945"
    assert location_edge.relation == EdgeRelation.REFERENCES
    assert location_edge.source == EdgeSource.SOURCE


def test_instagram_archive_adapter_filters_by_entity_type_and_date(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "posts_1.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Old post",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                },
                {
                    "creation_timestamp": 1712260800,
                    "title": "New post",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                },
            ]
        ),
        encoding="utf-8",
    )

    (content_dir / "stories.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "media": [{"uri": "story.jpg", "media_type": "photo"}],
                }
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

    adapter = InstagramArchiveAdapter(path=str(tmp_path))

    # Test entity type filtering
    posts_only = adapter.ingest(entity_types=["post"])
    assert len(posts_only.units) == 2
    assert all(u.source_entity_type == "post" for u in posts_only.units)

    stories_only = adapter.ingest(entity_types=["story"])
    assert len(stories_only.units) == 1
    assert all(u.source_entity_type == "story" for u in stories_only.units)

    messages_only = adapter.ingest(entity_types=["message"])
    assert len(messages_only.units) == 1
    assert all(u.source_entity_type == "message" for u in messages_only.units)

    # Test date filtering
    filtered = adapter.ingest(
        since=SyncState(
            source_project="instagram_archive",
            source_entity_type="post",
            last_sync_at=datetime(2024, 4, 2, tzinfo=timezone.utc),
        )
    )
    assert len(filtered.units) == 1  # Only new post (4/5) is included


def test_instagram_archive_adapter_handles_multiple_media_types(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "posts_1.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Album post",
                    "media": [
                        {
                            "uri": "photo1.jpg",
                            "media_type": "photo",
                        },
                        {
                            "uri": "video1.mp4",
                            "media_type": "video",
                        },
                        {
                            "uri": "photo2.jpg",
                            "media_type": "photo",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert len(unit.metadata["media"]) == 3
    assert unit.metadata["media"][0]["media_url"] == "photo1.jpg"
    assert unit.metadata["media"][0]["media_type"] == "photo"
    assert unit.metadata["media"][1]["media_url"] == "video1.mp4"
    assert unit.metadata["media"][1]["media_type"] == "video"
    assert "Media: 3 items (photo, video, photo)" in unit.content


def test_instagram_archive_adapter_handles_empty_and_invalid_files(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "posts_1.json").write_text("[]", encoding="utf-8")
    (content_dir / "posts_2.json").write_text("not json", encoding="utf-8")

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_instagram_archive_adapter_nonexistent_path():
    result = InstagramArchiveAdapter(path="/nonexistent/path").ingest()

    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_instagram_archive_adapter_handles_alternative_file_locations(tmp_path):
    # Test posts at root level (not in content/)
    (tmp_path / "posts_1.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "Root level post",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    # Test stories at root level
    (tmp_path / "stories.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "media": [{"uri": "story.jpg", "media_type": "photo"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    entity_types = {u.source_entity_type for u in result.units}
    assert entity_types == {"post", "story"}


def test_instagram_archive_adapter_handles_posts_without_location(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "posts_1.json").write_text(
        json.dumps(
            [
                {
                    "creation_timestamp": 1712001600,
                    "title": "No location post",
                    "media": [{"uri": "photo.jpg", "media_type": "photo"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 0  # No location edges
    assert "location" not in result.units[0].metadata


def test_instagram_archive_adapter_handles_messages_without_media(tmp_path):
    messages_dir = tmp_path / "messages" / "inbox" / "conv1"
    messages_dir.mkdir(parents=True)
    (messages_dir / "message_1.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "sender_name": "Alice",
                        "timestamp_ms": 1712001600000,
                        "content": "Just text, no media",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = InstagramArchiveAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert "media_url" not in unit.metadata
    assert "media_type" not in unit.metadata
    assert "Media" not in unit.content
