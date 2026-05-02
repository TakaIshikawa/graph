from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.atom import AtomAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_atom_ingests_entries_with_metadata_and_deduplicated_tags(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Example Feed</title>
          <entry>
            <id>tag:example.com,2026:entry-1</id>
            <title>Atom entry</title>
            <link href="https://example.com/entry-1" rel="alternate" type="text/html"/>
            <link href="https://example.com/entry-1/comments" rel="replies"/>
            <updated>2026-04-24T12:00:00Z</updated>
            <published>2026-04-23T10:30:00Z</published>
            <author><name>Ada</name></author>
            <category term="Research"/>
            <category term="#research"/>
            <category label="Agents"/>
            <summary><![CDATA[<p>Summary <strong>text</strong>.</p>]]></summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = AtomAdapter(path=str(feed)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "atom"
    assert unit.source_entity_type == "atom_entry"
    assert unit.source_id == "tag:example.com,2026:entry-1"
    assert unit.title == "Atom entry"
    assert unit.content == "Summary text."
    assert unit.content_type == "artifact"
    assert unit.tags == ["Research", "Agents"]
    assert unit.created_at == datetime(2026, 4, 23, 10, 30, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    assert unit.metadata["id"] == "tag:example.com,2026:entry-1"
    assert unit.metadata["link"] == "https://example.com/entry-1"
    assert unit.metadata["link_hrefs"] == [
        "https://example.com/entry-1",
        "https://example.com/entry-1/comments",
    ]
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["published"] == "2026-04-23T10:30:00Z"
    assert unit.metadata["updated"] == "2026-04-24T12:00:00Z"
    assert unit.metadata["categories"] == ["Research", "Agents"]


def test_atom_uses_link_as_source_id_when_id_is_missing(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Linked entry</title>
            <link href="https://example.com/linked"/>
            <content>Content body.</content>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = AtomAdapter(path=str(feed)).ingest()

    assert result.units[0].source_id == "https://example.com/linked"
    assert result.units[0].content == "Content body."


def test_atom_reads_all_xml_files_under_directory_and_skips_malformed_xml(tmp_path):
    first = tmp_path / "first.xml"
    first.write_text(
        """<feed xmlns="http://www.w3.org/2005/Atom">
          <entry><id>first</id><title>First</title><summary>One</summary></entry>
        </feed>
        """,
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.xml"
    second.write_text(
        """<feed xmlns="http://www.w3.org/2005/Atom">
          <entry><id>second</id><title>Second</title><summary>Two</summary></entry>
        </feed>
        """,
        encoding="utf-8",
    )
    broken = tmp_path / "broken.xml"
    broken.write_text("<feed><entry>", encoding="utf-8")

    result = AtomAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["first", "second"]


def test_atom_since_filter_uses_updated_then_published_timestamps(tmp_path):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>old</id><title>Old</title>
            <updated>2026-04-20T00:00:00Z</updated>
            <summary>Old summary</summary>
          </entry>
          <entry>
            <id>new</id><title>New</title>
            <published>2026-04-22T00:00:00Z</published>
            <summary>New summary</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )

    result = AtomAdapter(path=str(feed)).ingest(
        since=SyncState(
            source_project="atom",
            source_entity_type="atom_entry",
            last_sync_at=datetime(2026, 4, 21, tzinfo=timezone.utc),
        )
    )
    skipped = AtomAdapter(path=str(feed)).ingest(entity_types=["feed_item"])

    assert [unit.source_id for unit in result.units] == ["new"]
    assert skipped.units == []
    assert skipped.edges == []


def test_atom_adapter_is_registered():
    assert "atom" in list_adapters()
    adapter = get_adapter("atom", path="/tmp/feed.xml")
    assert isinstance(adapter, AtomAdapter)
    assert adapter.name == "atom"
