from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from graph.adapters.plain_text import PlainTextAdapter
from graph.types.enums import ContentType, SourceProject


def test_ingests_single_text_file(tmp_path):
    """Test ingesting a single plain text file."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is a plain text note.\nWith multiple lines.\n", encoding="utf-8")

    result = PlainTextAdapter(path=str(text_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    expected_digest = hashlib.sha256("notes.txt".encode("utf-8")).hexdigest()[:16]
    assert unit.source_project == SourceProject.PLAIN_TEXT
    assert unit.source_entity_type == "plain_text"
    assert unit.source_id == f"plain_text:{expected_digest}"
    assert unit.title == "notes"
    assert unit.content == "This is a plain text note.\nWith multiple lines.\n"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.metadata == {
        "source_file": "notes.txt",
    }
    assert isinstance(unit.created_at, datetime)
    assert isinstance(unit.updated_at, datetime)


def test_ingests_multiple_text_files(tmp_path):
    """Test ingesting multiple plain text files from a directory."""
    dir1 = tmp_path / "documents"
    dir1.mkdir()

    file1 = dir1 / "file1.txt"
    file1.write_text("Content of file 1", encoding="utf-8")

    file2 = dir1 / "file2.txt"
    file2.write_text("Content of file 2", encoding="utf-8")

    dir2 = dir1 / "subdir"
    dir2.mkdir()
    file3 = dir2 / "file3.txt"
    file3.write_text("Content of file 3", encoding="utf-8")

    result = PlainTextAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 3

    # Files should be sorted
    assert result.units[0].title == "file1"
    assert result.units[0].content == "Content of file 1"
    assert result.units[0].metadata["source_file"] == "file1.txt"

    assert result.units[1].title == "file2"
    assert result.units[1].content == "Content of file 2"
    assert result.units[1].metadata["source_file"] == "file2.txt"

    assert result.units[2].title == "file3"
    assert result.units[2].content == "Content of file 3"
    assert result.units[2].metadata["source_file"] == "subdir/file3.txt"


def test_ingests_empty_text_file(tmp_path):
    """Test ingesting an empty plain text file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    result = PlainTextAdapter(path=str(empty_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "empty"
    assert unit.content == ""
    assert unit.metadata["source_file"] == "empty.txt"


def test_handles_unicode_content(tmp_path):
    """Test handling Unicode characters in plain text files."""
    unicode_file = tmp_path / "unicode.txt"
    unicode_file.write_text("こんにちは世界\n你好世界\nПривет мир\n🌍🌎🌏", encoding="utf-8")

    result = PlainTextAdapter(path=str(unicode_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "unicode"
    assert unit.content == "こんにちは世界\n你好世界\nПривет мир\n🌍🌎🌏"


def test_skips_non_txt_files(tmp_path):
    """Test that non-.txt files are ignored."""
    dir1 = tmp_path / "mixed"
    dir1.mkdir()

    txt_file = dir1 / "document.txt"
    txt_file.write_text("Text content", encoding="utf-8")

    md_file = dir1 / "readme.md"
    md_file.write_text("Markdown content", encoding="utf-8")

    log_file = dir1 / "app.log"
    log_file.write_text("Log content", encoding="utf-8")

    result = PlainTextAdapter(path=str(dir1)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "document"


def test_nonexistent_path_returns_empty_result(tmp_path):
    """Test that a nonexistent path returns an empty result."""
    nonexistent = tmp_path / "nonexistent"

    result = PlainTextAdapter(path=str(nonexistent)).ingest()

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_returns_empty_result(tmp_path):
    """Test that filtering by a different entity type returns empty result."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Content", encoding="utf-8")

    result = PlainTextAdapter(path=str(tmp_path)).ingest(entity_types=["other_type"])

    assert result.units == []
    assert result.edges == []


def test_entity_type_filter_includes_plain_text(tmp_path):
    """Test that filtering by plain_text entity type includes results."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Content", encoding="utf-8")

    result = PlainTextAdapter(path=str(tmp_path)).ingest(entity_types=["plain_text"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "plain_text"


def test_sync_state_filters_unmodified_files(tmp_path):
    """Test that sync state filtering skips unmodified files."""
    from graph.types.models import SyncState

    old_file = tmp_path / "old.txt"
    old_file.write_text("Old content", encoding="utf-8")

    # Get the file's timestamp
    old_mtime = old_file.stat().st_mtime

    # Create a sync state with a timestamp after the file modification
    sync_state = SyncState(
        source_project="plain_text",
        source_entity_type="plain_text",
        last_sync_at=datetime.fromtimestamp(old_mtime + 1, tz=timezone.utc),
    )

    result = PlainTextAdapter(path=str(tmp_path)).ingest(since=sync_state)

    # File should be filtered out since it wasn't modified after sync
    assert result.units == []


def test_uses_source_id_root_for_relative_paths(tmp_path):
    """Test that source_id_root is used for computing relative paths."""
    subdir = tmp_path / "project" / "docs"
    subdir.mkdir(parents=True)

    text_file = subdir / "note.txt"
    text_file.write_text("Content", encoding="utf-8")

    result = PlainTextAdapter(
        path=str(text_file),
        source_id_root=str(tmp_path)
    ).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["source_file"] == "project/docs/note.txt"


def test_root_path_parameter_works(tmp_path):
    """Test that root_path parameter works as an alternative to path."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Content", encoding="utf-8")

    result = PlainTextAdapter(root_path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "test"


def test_filename_with_special_characters(tmp_path):
    """Test handling filenames with special characters."""
    special_file = tmp_path / "my-file_name (v2).txt"
    special_file.write_text("Content", encoding="utf-8")

    result = PlainTextAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "my-file_name (v2)"
    assert result.units[0].metadata["source_file"] == "my-file_name (v2).txt"


def test_handles_replacement_characters_for_invalid_utf8(tmp_path):
    """Test that invalid UTF-8 sequences are replaced."""
    invalid_file = tmp_path / "invalid.txt"
    # Write some invalid UTF-8 bytes
    invalid_file.write_bytes(b"Valid text\xFF\xFEInvalid bytes\xC0\xC1")

    result = PlainTextAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    # The exact replacement character may vary, but it should not crash
    assert len(result.units[0].content) > 0
    assert "Valid text" in result.units[0].content
