from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.bookmarks_html import BookmarksHtmlAdapter
from graph.types.enums import ContentType, SourceProject


def test_bookmarks_html_adapter_ingests_nested_bookmarks_and_metadata(tmp_path):
    bookmarks = tmp_path / "bookmarks.html"
    bookmarks.write_text(
        """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
        <TITLE>Bookmarks</TITLE>
        <H1>Bookmarks</H1>
        <DL><p>
          <DT><H3 ADD_DATE="1713949200" LAST_MODIFIED="1713949300">Bookmarks Bar</H3>
          <DL><p>
            <DT><H3>Research</H3>
            <DL><p>
              <DT><A HREF="https://example.com/agent-eval?ref=bookmarks"
                     ADD_DATE="1713952800"
                     LAST_MODIFIED="1713956400">Agent &amp; Evaluation</A>
              <DT><A HREF="https://example.org/no-metadata">No Metadata</A>
            </DL><p>
            <DT><A HREF="https://example.net/top" ADD_DATE="1713960000">Top Level</A>
          </DL><p>
        </DL><p>
        """,
        encoding="utf-8",
    )

    result = BookmarksHtmlAdapter(path=str(bookmarks)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "https://example.com/agent-eval?ref=bookmarks",
        "https://example.org/no-metadata",
        "https://example.net/top",
    ]

    first = result.units[0]
    assert first.source_project == SourceProject.BOOKMARKS_HTML
    assert first.source_entity_type == "bookmark"
    assert first.title == "Agent & Evaluation"
    assert first.content == (
        "Agent & Evaluation\n"
        "https://example.com/agent-eval?ref=bookmarks\n"
        "Bookmarks Bar/Research"
    )
    assert first.content_type == ContentType.ARTIFACT
    assert first.tags == ["Bookmarks Bar", "Bookmarks Bar/Research"]
    assert first.metadata == {
        "url": "https://example.com/agent-eval?ref=bookmarks",
        "folder_path": "Bookmarks Bar/Research",
        "add_date": "1713952800",
        "last_modified": "1713956400",
    }
    assert first.created_at == datetime.fromtimestamp(1713952800, tz=timezone.utc)
    assert first.updated_at == datetime.fromtimestamp(1713956400, tz=timezone.utc)

    missing = result.units[1]
    assert missing.source_project == SourceProject.BOOKMARKS_HTML
    assert missing.metadata["url"] == "https://example.org/no-metadata"
    assert missing.metadata["add_date"] == ""
    assert missing.metadata["last_modified"] == ""
    assert missing.tags == ["Bookmarks Bar", "Bookmarks Bar/Research"]
    assert missing.created_at.tzinfo == timezone.utc
    assert missing.updated_at.tzinfo == timezone.utc

    top_level = result.units[2]
    assert top_level.metadata["folder_path"] == "Bookmarks Bar"
    assert top_level.tags == ["Bookmarks Bar"]
    assert top_level.created_at == datetime.fromtimestamp(1713960000, tz=timezone.utc)
    assert top_level.updated_at == datetime.fromtimestamp(1713960000, tz=timezone.utc)
    assert result.edges == []


def test_bookmarks_html_adapter_filters_and_missing_path(tmp_path):
    missing_path = tmp_path / "missing.html"
    assert BookmarksHtmlAdapter(path=str(missing_path)).ingest().units == []

    bookmarks = tmp_path / "bookmarks.html"
    bookmarks.write_text(
        """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <DL><p>
          <DT><A HREF="https://example.com" ADD_DATE="1713952800">Example</A>
        </DL><p>
        """,
        encoding="utf-8",
    )

    result = BookmarksHtmlAdapter(path=str(bookmarks)).ingest(entity_types=["feed_item"])

    assert result.units == []
    assert result.edges == []
