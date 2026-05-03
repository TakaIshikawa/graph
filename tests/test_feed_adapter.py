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
    assert unit.source_project == SourceProject.FEED
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


def test_feed_ingests_atom_entries_with_metadata(tmp_path):
    """Test ingesting Atom feed with full metadata."""
    feed = tmp_path / "atom.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Example Atom Feed</title>
          <entry>
            <id>urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a</id>
            <title>Atom Entry Title</title>
            <link href="https://example.com/atom-entry-1" rel="alternate"/>
            <published>2026-04-15T10:30:00Z</published>
            <updated>2026-04-16T12:00:00Z</updated>
            <author>
              <name>Jane Doe</name>
            </author>
            <category term="Technology" label="Tech"/>
            <category term="Science"/>
            <summary>This is the entry summary.</summary>
            <content type="html"><![CDATA[<p>Full entry <strong>content</strong> here.</p>]]></content>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FEED
    assert unit.source_entity_type == "feed_item"
    assert unit.title == "Atom Entry Title"
    assert unit.content == "Full entry content here."
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["Technology", "Science"]
    assert unit.created_at == datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc)
    assert unit.metadata["feed_title"] == "Example Atom Feed"
    assert unit.metadata["id"] == "urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a"
    assert unit.metadata["link"] == "https://example.com/atom-entry-1"
    assert unit.metadata["author"] == "Jane Doe"
    assert unit.metadata["published"] == "2026-04-15T10:30:00Z"
    assert unit.metadata["updated"] == "2026-04-16T12:00:00Z"


def test_feed_ingests_atom_entry_with_summary_only(tmp_path):
    """Test ingesting Atom entry that only has summary (no content)."""
    feed = tmp_path / "atom-summary.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Summary Feed</title>
          <entry>
            <id>urn:uuid:2225c695-cfb8-4ebb-aaaa-80da344efa6b</id>
            <title>Entry with Summary</title>
            <link href="https://example.com/entry-2"/>
            <published>2026-05-01T08:00:00Z</published>
            <summary type="text">This is just a summary without full content.</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Entry with Summary"
    assert unit.content == "This is just a summary without full content."
    assert unit.metadata["link"] == "https://example.com/entry-2"


def test_feed_ingests_multiple_atom_entries(tmp_path):
    """Test ingesting multiple entries from a single Atom feed."""
    feed = tmp_path / "multi-atom.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Multi Entry Feed</title>
          <entry>
            <id>entry-1</id>
            <title>First Entry</title>
            <link href="https://example.com/1"/>
            <published>2026-05-01T10:00:00Z</published>
            <summary>First entry summary</summary>
          </entry>
          <entry>
            <id>entry-2</id>
            <title>Second Entry</title>
            <link href="https://example.com/2"/>
            <published>2026-05-02T10:00:00Z</published>
            <summary>Second entry summary</summary>
          </entry>
          <entry>
            <id>entry-3</id>
            <title>Third Entry</title>
            <link href="https://example.com/3"/>
            <published>2026-05-03T10:00:00Z</published>
            <summary>Third entry summary</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 3
    titles = [unit.title for unit in result.units]
    assert "First Entry" in titles
    assert "Second Entry" in titles
    assert "Third Entry" in titles
    ids = [unit.metadata["id"] for unit in result.units]
    assert "entry-1" in ids
    assert "entry-2" in ids
    assert "entry-3" in ids


def test_feed_handles_atom_with_html_content(tmp_path):
    """Test that HTML content in Atom feeds is properly cleaned."""
    feed = tmp_path / "html-atom.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>HTML Content Feed</title>
          <entry>
            <id>html-entry</id>
            <title>Entry with HTML</title>
            <link href="https://example.com/html"/>
            <published>2026-05-01T10:00:00Z</published>
            <content type="html">
              <![CDATA[
                <div>
                  <h1>Heading</h1>
                  <p>Paragraph with <em>emphasis</em> and <strong>bold</strong>.</p>
                  <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                  </ul>
                </div>
              ]]>
            </content>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # HTML should be cleaned to plain text
    assert "<" not in unit.content
    assert ">" not in unit.content
    assert "Heading" in unit.content
    assert "Paragraph" in unit.content
    assert "emphasis" in unit.content
    assert "bold" in unit.content


def test_feed_handles_multiple_categories_in_atom(tmp_path):
    """Test extraction of multiple categories from Atom feed."""
    feed = tmp_path / "categories-atom.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Categorized Feed</title>
          <entry>
            <id>categorized-entry</id>
            <title>Entry with Categories</title>
            <link href="https://example.com/cat"/>
            <published>2026-05-01T10:00:00Z</published>
            <category term="technology"/>
            <category term="programming" label="Programming"/>
            <category term="python"/>
            <summary>Entry with multiple categories</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert len(unit.tags) == 3
    assert "technology" in unit.tags
    assert "programming" in unit.tags
    assert "python" in unit.tags


def test_feed_handles_atom_alternate_links(tmp_path):
    """Test that Atom alternate links are properly extracted."""
    feed = tmp_path / "links-atom.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Links Feed</title>
          <entry>
            <id>links-entry</id>
            <title>Entry with Multiple Links</title>
            <link href="https://example.com/self" rel="self"/>
            <link href="https://example.com/alternate" rel="alternate"/>
            <link href="https://example.com/related" rel="related"/>
            <published>2026-05-01T10:00:00Z</published>
            <summary>Entry with multiple link types</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should prefer alternate link
    assert unit.metadata["link"] == "https://example.com/alternate"


def test_feed_handles_rss_content_encoded(tmp_path):
    """Test RSS feed with content:encoded field."""
    feed = tmp_path / "content-encoded.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
          <channel>
            <title>Content Encoded Feed</title>
            <item>
              <guid>encoded-1</guid>
              <title>Entry with Encoded Content</title>
              <link>https://example.com/encoded-1</link>
              <pubDate>Wed, 01 May 2026 10:00:00 +0000</pubDate>
              <description>Short description</description>
              <content:encoded><![CDATA[<p>Full HTML content with <strong>formatting</strong>.</p>]]></content:encoded>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should prefer content:encoded over description
    assert unit.content == "Full HTML content with formatting ."
    assert "<" not in unit.content


def test_feed_handles_rss_dc_creator(tmp_path):
    """Test RSS feed with dc:creator field for author."""
    feed = tmp_path / "dc-creator.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
          <channel>
            <title>DC Creator Feed</title>
            <item>
              <guid>dc-1</guid>
              <title>Entry with DC Creator</title>
              <link>https://example.com/dc-1</link>
              <pubDate>Wed, 01 May 2026 10:00:00 +0000</pubDate>
              <dc:creator>John Doe</dc:creator>
              <description>Entry description</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["author"] == "John Doe"


def test_feed_handles_multiple_rss_categories(tmp_path):
    """Test RSS feed with multiple category tags."""
    feed = tmp_path / "rss-categories.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Categories Feed</title>
            <item>
              <guid>cat-1</guid>
              <title>Entry with Categories</title>
              <link>https://example.com/cat-1</link>
              <pubDate>Wed, 01 May 2026 10:00:00 +0000</pubDate>
              <category>News</category>
              <category>Technology</category>
              <category>AI</category>
              <description>Entry with multiple categories</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert len(unit.tags) == 3
    assert "News" in unit.tags
    assert "Technology" in unit.tags
    assert "AI" in unit.tags


def test_feed_filters_by_entity_type(tmp_path):
    """Test that entity type filtering works correctly."""
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <guid>test-1</guid>
              <title>Test Item</title>
              <description>Test description</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    # Should return empty when filtering by different entity type
    result = FeedAdapter(sources=str(feed)).ingest(entity_types=["other_type"])
    assert len(result.units) == 0

    # Should return results when filtering by feed_item
    result = FeedAdapter(sources=str(feed)).ingest(entity_types=["feed_item"])
    assert len(result.units) == 1


def test_feed_handles_directory_of_xml_files(tmp_path):
    """Test ingesting multiple XML files from a directory."""
    feed_dir = tmp_path / "feeds"
    feed_dir.mkdir()

    feed1 = feed_dir / "feed1.xml"
    feed1.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Feed 1</title>
            <item>
              <guid>feed1-item1</guid>
              <title>Feed 1 Item 1</title>
              <description>Description 1</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    feed2 = feed_dir / "feed2.xml"
    feed2.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Feed 2</title>
          <entry>
            <id>feed2-entry1</id>
            <title>Feed 2 Entry 1</title>
            <summary>Summary 2</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = FeedAdapter(sources=str(feed_dir)).ingest()

    assert len(result.units) == 2
    titles = [unit.title for unit in result.units]
    assert "Feed 1 Item 1" in titles
    assert "Feed 2 Entry 1" in titles


def test_feed_filters_by_sync_state(tmp_path):
    """Test that sync state filtering works correctly."""
    from graph.types.models import SyncState

    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <guid>old-item</guid>
              <title>Old Item</title>
              <pubDate>Wed, 01 Jan 2020 10:00:00 +0000</pubDate>
              <description>Old item</description>
            </item>
            <item>
              <guid>new-item</guid>
              <title>New Item</title>
              <pubDate>Wed, 01 May 2026 10:00:00 +0000</pubDate>
              <description>New item</description>
            </item>
          </channel>
        </rss>
        """,
        encoding="utf-8",
    )

    # Filter with sync state after the old item but before the new item
    sync_state = SyncState(
        source_project="feed",
        source_entity_type="feed_item",
        last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    result = FeedAdapter(sources=str(feed)).ingest(since=sync_state)

    # Should only get the new item
    assert len(result.units) == 1
    assert result.units[0].title == "New Item"
