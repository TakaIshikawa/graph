from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.twitter_archive import TwitterArchiveAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_twitter_archive_adapter_ingests_plain_json_metadata_and_edges(tmp_path):
    archive_path = tmp_path / "tweets.json"
    archive_path.write_text(
        json.dumps(
            [
                {
                    "tweet": {
                        "id_str": "100",
                        "created_at": "Tue Apr 01 10:00:00 +0000 2026",
                        "full_text": "First archive note with #Graph and @alice https://t.co/a",
                        "screen_name": "taka",
                        "entities": {
                            "hashtags": [{"text": "Graph"}],
                            "user_mentions": [
                                {"screen_name": "alice", "name": "Alice A.", "id_str": "user-1"}
                            ],
                            "urls": [
                                {
                                    "url": "https://t.co/a",
                                    "expanded_url": "https://example.com/a",
                                    "display_url": "example.com/a",
                                }
                            ],
                        },
                        "favorite_count": "2",
                        "retweet_count": "1",
                        "source": '<a href="https://mobile.twitter.com">Twitter Web App</a>',
                        "lang": "en",
                    }
                },
                {
                    "tweet": {
                        "id_str": "101",
                        "created_at": "2026-04-01T10:05:00Z",
                        "full_text": "Replying with quote context.",
                        "screen_name": "taka",
                        "in_reply_to_status_id_str": "100",
                        "quoted_status_id_str": "102",
                        "in_reply_to_screen_name": "taka",
                    }
                },
                {
                    "tweet": {
                        "id_str": "102",
                        "created_at": "2026-04-01T10:06:00Z",
                        "full_text": "Quoted note.",
                        "screen_name": "taka",
                        "retweeted_status_id_str": "77",
                    }
                },
                {
                    "tweet": {
                        "id_str": "103",
                        "created_at": "2026-04-01T10:07:00Z",
                        "full_text": "   ",
                    }
                },
            ]
        ),
        encoding="utf-8",
    )

    result = TwitterArchiveAdapter(path=str(archive_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "twitter_archive:100",
        "twitter_archive:101",
        "twitter_archive:102",
    ]
    first = result.units[0]
    assert first.source_project == SourceProject.TWITTER_ARCHIVE
    assert first.source_entity_type == "tweet"
    assert first.title == "@taka 2026-04-01 First archive note with #Graph and @alice https://t.co/a"
    assert first.content == "First archive note with #Graph and @alice https://t.co/a"
    assert first.created_at == datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    assert first.updated_at == datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    assert first.tags == ["twitter", "hashtag-graph", "mention-alice"]
    assert first.metadata["tweet_id"] == "100"
    assert first.metadata["url"] == "https://twitter.com/taka/status/100"
    assert first.metadata["hashtags"] == ["graph"]
    assert first.metadata["mentions"] == [
        {"screen_name": "alice", "name": "Alice A.", "id": "user-1"}
    ]
    assert first.metadata["urls"] == [
        {
            "url": "https://t.co/a",
            "expanded_url": "https://example.com/a",
            "display_url": "example.com/a",
        }
    ]
    assert first.metadata["favorite_count"] == 2
    assert first.metadata["retweet_count"] == 1
    assert first.metadata["source_path"] == "tweets.json"
    assert result.units[1].metadata["references"] == [
        {"relation_type": "twitter_reply", "tweet_id": "100"},
        {"relation_type": "twitter_quote", "tweet_id": "102"},
    ]
    assert result.units[2].metadata["retweeted_tweet_id"] == "77"

    assert len(result.edges) == 2
    assert [(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges] == [
        ("twitter_archive:101", "twitter_archive:100", "twitter_reply"),
        ("twitter_archive:101", "twitter_archive:102", "twitter_quote"),
    ]
    assert all(edge.relation == EdgeRelation.REFERENCES for edge in result.edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in result.edges)
    assert all(edge.metadata["source_project"] == "twitter_archive" for edge in result.edges)


def test_twitter_archive_adapter_ingests_javascript_wrapped_archive_and_filters(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tweets.js").write_text(
        """
window.YTD.tweets.part0 = [
  {
    "tweet": {
      "id_str": "200",
      "created_at": "2026-04-01T10:00:00Z",
      "full_text": "Old tweet.",
      "entities": {"urls": []}
    }
  },
  {
    "tweet": {
      "id_str": "201",
      "created_at": "2026-04-03T10:00:00Z",
      "full_text": "New quote https://x.com/taka/status/200",
      "entities": {
        "urls": [
          {
            "url": "https://t.co/q",
            "expanded_url": "https://x.com/taka/status/200",
            "display_url": "x.com/taka/status/200"
          }
        ]
      }
    }
  }
];
""",
        encoding="utf-8",
    )
    adapter = TwitterArchiveAdapter(root_path=str(tmp_path))

    result = adapter.ingest()
    filtered = adapter.ingest(
        since=SyncState(
            source_project="twitter_archive",
            source_entity_type="tweet",
            last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )
    )
    excluded = adapter.ingest(entity_types=["other"])

    assert [unit.source_id for unit in result.units] == [
        "twitter_archive:200",
        "twitter_archive:201",
    ]
    assert len(result.edges) == 1
    assert result.edges[0].from_unit_id == "twitter_archive:201"
    assert result.edges[0].to_unit_id == "twitter_archive:200"
    assert result.edges[0].metadata["relation_type"] == "twitter_quote"
    assert result.units[0].metadata["source_path"] == "data/tweets.js"
    assert [unit.metadata["tweet_id"] for unit in filtered.units] == ["201"]
    assert filtered.edges == []
    assert excluded.units == []
    assert excluded.edges == []
