from __future__ import annotations

import json

from graph.adapters.twitter_bookmarks_json import TwitterBookmarksJsonAdapter


def test_twitter_bookmarks_json_handles_archive_wrappers_and_generates_urls(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text(
        json.dumps(
            {
                "bookmarks": [
                    {
                        "tweet": {
                            "id_str": "123",
                            "full_text": "Saved tweet",
                            "screen_name": "ada",
                            "created_at": "2026-05-01T12:00:00Z",
                            "conversation_id_str": "100",
                            "entities": {"media": [{"media_url_https": "https://img.test/a.jpg"}]},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = TwitterBookmarksJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "twitter_bookmarks_json:tweet:123"
    assert unit.metadata["author_handle"] == "ada"
    assert unit.metadata["url"] == "https://twitter.com/ada/status/123"
    assert unit.metadata["media_urls"] == ["https://img.test/a.jpg"]
    assert unit.metadata["conversation_id"] == "100"


def test_twitter_bookmarks_json_handles_plain_arrays(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text(json.dumps([{"tweet_id": "456", "text": "Plain", "author_handle": "@grace"}]), encoding="utf-8")

    unit = TwitterBookmarksJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["url"] == "https://twitter.com/grace/status/456"
