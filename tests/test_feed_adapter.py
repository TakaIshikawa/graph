from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.feed import FeedAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_feed_ingests_rss_items_with_metadata(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Example Feed</title>
            <item>
              <guid>entry-1</guid>
              <title>RSS item</title>
              <link>https://example.com/entry-1</link>
              <pubDate>Wed, 23 Apr 2026 10:30:00 +0000</pubDate>
              <author>ada@example.com</author>
              <category>Research</category>
              <description><![CDATA[<p>Summary <strong>text</strong>.</p>]]></description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.ME
    assert unit.source_entity_type == "feed_item"
    assert unit.title == "RSS item"
    assert unit.content == "Summary text ."
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["Research"]
    assert unit.created_at == datetime(2026, 4, 23, 10, 30, tzinfo=timezone.utc)
    assert unit.metadata["feed_title"] == "Example Feed"
    assert unit.metadata["id"] == "entry-1"
    assert unit.metadata["link"] == "https://example.com/entry-1"
    assert unit.metadata["author"] == "ada@example.com"
    assert unit.metadata["published"] == "Wed, 23 Apr 2026 10:30:00 +0000"


def test_feed_captures_multiple_enclosures_with_metadata(tmp_path):
    feed = tmp_path / "podcast.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Podcast Feed</title>
            <item>
              <guid>episode-1</guid>
              <title>Episode One</title>
              <link>https://example.com/episode-1</link>
              <pubDate>Thu, 01 May 2026 12:00:00 +0000</pubDate>
              <description>Episode description here.</description>
              <enclosure url="https://example.com/audio.mp3" type="audio/mpeg" length="12345678"/>
              <enclosure url="https://example.com/video.mp4" type="video/mp4" length="98765432"/>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Episode One"
    assert unit.content == "Episode description here."
    assert "enclosures" in unit.metadata
    assert len(unit.metadata["enclosures"]) == 2
    assert unit.metadata["enclosures"][0] == {
        "url": "https://example.com/audio.mp3",
        "type": "audio/mpeg",
        "length": "12345678",
    }
    assert unit.metadata["enclosures"][1] == {
        "url": "https://example.com/video.mp4",
        "type": "video/mp4",
        "length": "98765432",
    }


def test_feed_includes_enclosure_url_in_content_when_no_description(tmp_path):
    feed = tmp_path / "podcast.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Podcast Feed</title>
            <item>
              <guid>episode-2</guid>
              <title>Episode Two</title>
              <link>https://example.com/episode-2</link>
              <pubDate>Fri, 02 May 2026 12:00:00 +0000</pubDate>
              <enclosure url="https://example.com/audio-ep2.mp3" type="audio/mpeg" length="11111111"/>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Episode Two"
    # Content should include title + enclosure URL when no description
    assert unit.content == "Episode Two https://example.com/audio-ep2.mp3"
    assert "enclosures" in unit.metadata
    assert len(unit.metadata["enclosures"]) == 1
    assert unit.metadata["enclosures"][0]["url"] == "https://example.com/audio-ep2.mp3"


def test_feed_no_enclosures_metadata_when_none_present(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Simple Feed</title>
            <item>
              <guid>entry-3</guid>
              <title>Simple entry</title>
              <description>Just a simple description.</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.content == "Just a simple description."
    assert "enclosures" not in unit.metadata


def test_feed_handles_enclosure_with_partial_attributes(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Partial Feed</title>
            <item>
              <guid>entry-4</guid>
              <title>Partial enclosure</title>
              <description>Has partial enclosure data.</description>
              <enclosure url="https://example.com/file.pdf"/>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert "enclosures" in unit.metadata
    assert len(unit.metadata["enclosures"]) == 1
    # Should only include attributes that are present
    assert unit.metadata["enclosures"][0] == {
        "url": "https://example.com/file.pdf",
    }


def test_feed_adapter_is_registered():
    assert "feed" in list_adapters()
    adapter = get_adapter("feed", sources="https://example.com/feed.xml")
    assert isinstance(adapter, FeedAdapter)
    assert adapter.name == "feed"
