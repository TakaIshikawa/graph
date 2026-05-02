"""Tests for MediaWiki XML dump ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.mediawiki import MediaWikiAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


MEDIAWIKI_DUMP = """<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Project Notes</title>
    <ns>0</ns>
    <id>10</id>
    <revision>
      <id>100</id>
      <timestamp>2025-01-01T10:00:00Z</timestamp>
      <contributor>
        <username>Ada</username>
        <id>7</id>
      </contributor>
      <text xml:space="preserve">Old text</text>
    </revision>
    <revision>
      <id>101</id>
      <timestamp>2025-01-02T11:12:13Z</timestamp>
      <contributor>
        <username>Ada</username>
        <id>7</id>
      </contributor>
      <text xml:space="preserve">Current page text.

[[Category:Research notes]]
[[Category:Research_notes|P]]
[[category:Knowledge Graph]]
</text>
    </revision>
  </page>
  <page>
    <title>Alias Page</title>
    <ns>0</ns>
    <id>11</id>
    <redirect title="Project Notes" />
    <revision>
      <id>201</id>
      <timestamp>2025-01-03T00:00:00Z</timestamp>
      <contributor>
        <ip>192.0.2.1</ip>
      </contributor>
      <text xml:space="preserve">#REDIRECT [[Project Notes]]</text>
    </revision>
  </page>
  <page>
    <title>Empty Page</title>
    <ns>0</ns>
    <id>12</id>
    <revision>
      <id>301</id>
      <timestamp>2025-01-04T00:00:00Z</timestamp>
      <text xml:space="preserve">   </text>
    </revision>
  </page>
  <page>
    <title>Deleted Page</title>
    <ns>0</ns>
    <id>13</id>
    <revision>
      <id>401</id>
      <timestamp>2025-01-05T00:00:00Z</timestamp>
      <text deleted="deleted" />
    </revision>
  </page>
</mediawiki>
"""


def test_mediawiki_parses_pages_and_metadata(tmp_path):
    dump = tmp_path / "wiki.xml"
    dump.write_text(MEDIAWIKI_DUMP, encoding="utf-8")

    result = MediaWikiAdapter(path=str(dump)).ingest()

    assert len(result.units) == 2
    page = result.units[0]
    redirect = result.units[1]

    assert page.source_project == SourceProject.MEDIAWIKI
    assert page.source_entity_type == "mediawiki_page"
    assert page.source_id == "mediawiki:10:101"
    assert page.title == "Project Notes"
    assert page.content == (
        "Current page text.\n\n"
        "[[Category:Research notes]]\n"
        "[[Category:Research_notes|P]]\n"
        "[[category:Knowledge Graph]]\n"
    )
    assert page.content_type == ContentType.ARTIFACT
    assert page.tags == ["Research notes", "Knowledge Graph"]
    assert page.created_at == datetime(2025, 1, 2, 11, 12, 13, tzinfo=timezone.utc)
    assert page.updated_at == datetime(2025, 1, 2, 11, 12, 13, tzinfo=timezone.utc)
    assert page.metadata == {
        "page_id": "10",
        "namespace": "0",
        "revision_id": "101",
        "contributor": "Ada",
        "timestamp": "2025-01-02T11:12:13Z",
        "redirect_target": "",
        "source_title": "Project Notes",
        "source_file": "wiki.xml",
        "contributor_id": "7",
    }

    assert redirect.source_id == "mediawiki:11:201"
    assert redirect.metadata["contributor"] == "192.0.2.1"
    assert redirect.metadata["redirect_target"] == "Project Notes"


def test_mediawiki_respects_entity_types_and_since(tmp_path):
    dump = tmp_path / "wiki.xml"
    dump.write_text(MEDIAWIKI_DUMP, encoding="utf-8")
    adapter = MediaWikiAdapter(path=str(dump))

    filtered = adapter.ingest(entity_types=["bookmark"])
    assert filtered.units == []
    assert filtered.edges == []

    since = SyncState(
        source_project="mediawiki",
        source_entity_type="mediawiki_page",
        last_sync_at=datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
    )
    result = adapter.ingest(entity_types=["mediawiki_page"], since=since)
    assert [unit.source_id for unit in result.units] == ["mediawiki:11:201"]


def test_mediawiki_reads_xml_files_from_directory(tmp_path):
    (tmp_path / "wiki.xml").write_text(MEDIAWIKI_DUMP, encoding="utf-8")
    (tmp_path / "ignored.txt").write_text(MEDIAWIKI_DUMP, encoding="utf-8")

    result = MediaWikiAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "mediawiki:10:101",
        "mediawiki:11:201",
    ]


def test_mediawiki_registry():
    assert "mediawiki" in list_adapters()
    adapter = get_adapter("mediawiki", path="/tmp/wiki.xml")
    assert isinstance(adapter, MediaWikiAdapter)
    assert adapter.name == "mediawiki"
