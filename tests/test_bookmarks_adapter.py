"""Tests for the Netscape bookmarks adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.bookmarks import BookmarksAdapter
from graph.types.enums import ContentType, SourceProject


def test_bookmarks_with_descriptions(tmp_path):
    """Test parsing bookmarks with DD description text."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        '    <DT><A HREF="https://example.com/doc" ADD_DATE="1609459200" LAST_MODIFIED="1609545600">Example Documentation</A>\n'
        "    <DD>This is a helpful documentation site with examples\n"
        '    <DT><A HREF="https://example.com/blog" ADD_DATE="1609459300">Example Blog</A>\n'
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 2

    # Bookmark with description
    unit_with_desc = result.units[0]
    assert unit_with_desc.source_project == SourceProject.BOOKMARKS
    assert unit_with_desc.source_entity_type == "bookmark"
    assert unit_with_desc.content_type == ContentType.ARTIFACT
    assert unit_with_desc.title == "Example Documentation"
    assert unit_with_desc.source_id == "https://example.com/doc"
    assert unit_with_desc.metadata["url"] == "https://example.com/doc"
    assert unit_with_desc.metadata["description"] == "This is a helpful documentation site with examples"
    assert "This is a helpful documentation site with examples" in unit_with_desc.content
    assert unit_with_desc.created_at == datetime(2021, 1, 1, tzinfo=timezone.utc)
    assert unit_with_desc.updated_at == datetime(2021, 1, 2, tzinfo=timezone.utc)

    # Bookmark without description
    unit_without_desc = result.units[1]
    assert unit_without_desc.title == "Example Blog"
    assert unit_without_desc.source_id == "https://example.com/blog"
    assert "description" not in unit_without_desc.metadata
    assert unit_without_desc.content == "Example Blog\nhttps://example.com/blog"


def test_bookmarks_with_nested_folders_and_descriptions(tmp_path):
    """Test parsing bookmarks in nested folders with descriptions."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        "    <DT><H3>Development</H3>\n"
        "    <DL><p>\n"
        "        <DT><H3>Python</H3>\n"
        "        <DL><p>\n"
        '            <DT><A HREF="https://docs.python.org" ADD_DATE="1609459200">Python Docs</A>\n'
        "            <DD>Official Python documentation\n"
        '            <DT><A HREF="https://pypi.org">PyPI</A>\n'
        "        </DL><p>\n"
        '        <DT><A HREF="https://github.com">GitHub</A>\n'
        "        <DD>Code hosting platform\n"
        "    </DL><p>\n"
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 3

    # Nested bookmark with description
    python_docs = result.units[0]
    assert python_docs.title == "Python Docs"
    assert python_docs.source_id == "https://docs.python.org"
    assert python_docs.metadata["folder_path"] == "Development/Python"
    assert python_docs.metadata["description"] == "Official Python documentation"
    assert "Official Python documentation" in python_docs.content
    assert python_docs.tags == ["Development", "Development/Python"]

    # Nested bookmark without description
    pypi = result.units[1]
    assert pypi.title == "PyPI"
    assert pypi.source_id == "https://pypi.org"
    assert pypi.metadata["folder_path"] == "Development/Python"
    assert "description" not in pypi.metadata
    assert pypi.tags == ["Development", "Development/Python"]

    # Bookmark in parent folder with description
    github = result.units[2]
    assert github.title == "GitHub"
    assert github.source_id == "https://github.com"
    assert github.metadata["folder_path"] == "Development"
    assert github.metadata["description"] == "Code hosting platform"
    assert "Code hosting platform" in github.content
    assert github.tags == ["Development"]


def test_bookmarks_multiline_description_normalized(tmp_path):
    """Test that multiline descriptions are normalized to single line."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        '    <DT><A HREF="https://example.com">Example Site</A>\n'
        "    <DD>This is a long\n"
        "        description that spans\n"
        "        multiple lines\n"
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["description"] == "This is a long description that spans multiple lines"


def test_bookmarks_empty_description_not_stored(tmp_path):
    """Test that empty DD tags don't create empty description metadata."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        '    <DT><A HREF="https://example.com">Example</A>\n'
        "    <DD>\n"
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert "description" not in unit.metadata


def test_bookmarks_description_only_associated_with_preceding_link(tmp_path):
    """Test that DD is only associated with the immediately preceding A tag."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        '    <DT><A HREF="https://first.com">First Link</A>\n'
        "    <DD>First description\n"
        '    <DT><A HREF="https://second.com">Second Link</A>\n'
        '    <DT><A HREF="https://third.com">Third Link</A>\n'
        "    <DD>Third description\n"
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 3

    # First bookmark has description
    assert result.units[0].title == "First Link"
    assert result.units[0].metadata["description"] == "First description"

    # Second bookmark has no description
    assert result.units[1].title == "Second Link"
    assert "description" not in result.units[1].metadata

    # Third bookmark has description
    assert result.units[2].title == "Third Link"
    assert result.units[2].metadata["description"] == "Third description"


def test_bookmarks_content_ordering(tmp_path):
    """Test that content includes title, URL, folder path, and description in correct order."""
    bookmarks_file = tmp_path / "bookmarks.html"
    bookmarks_file.write_text(
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>\n"
        "<HTML>\n"
        "<DL><p>\n"
        "    <DT><H3>Tools</H3>\n"
        "    <DL><p>\n"
        '        <DT><A HREF="https://example.com/tool">Example Tool</A>\n'
        "        <DD>A useful tool for developers\n"
        "    </DL><p>\n"
        "</DL>\n"
        "</HTML>\n",
        encoding="utf-8",
    )

    result = BookmarksAdapter(path=str(bookmarks_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]

    expected_content = "Example Tool\nhttps://example.com/tool\nTools\nA useful tool for developers"
    assert unit.content == expected_content
