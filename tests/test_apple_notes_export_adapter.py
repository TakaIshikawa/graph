from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_notes_export import AppleNotesExportAdapter
from graph.types.enums import SourceProject


def test_apple_notes_export_imports():
    """Test that Apple Notes export adapter can be instantiated."""
    adapter = AppleNotesExportAdapter()
    assert adapter.name == "apple_notes_export"
    assert adapter.entity_types == ["note"]


def test_apple_notes_export_parses_html(tmp_path):
    """Test parsing HTML notes."""
    note = tmp_path / "note.html"
    note.write_text("<html><head><title>My Note</title></head><body><p>Content</p></body></html>", encoding="utf-8")

    result = AppleNotesExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.APPLE_NOTES_EXPORT
    assert unit.source_entity_type == "note"
    assert unit.title == "My Note"


def test_apple_notes_export_imports_txt_with_front_matter_and_tags(tmp_path):
    note = tmp_path / "plain.txt"
    note.write_text(
        """---
title: Plain Note
created: 2025-01-01T00:00:00Z
modified: 2025-01-02T00:00:00Z
---
First line
#research
""",
        encoding="utf-8",
    )

    result = AppleNotesExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Plain Note"
    assert unit.content == "First line\n#research"
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert unit.tags == ["research"]


def test_apple_notes_export_reflects_nested_folder(tmp_path):
    folder = tmp_path / "Work"
    folder.mkdir()
    (folder / "note.txt").write_text("Nested note body", encoding="utf-8")

    result = AppleNotesExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["folder"] == "Work"
    assert result.units[0].tags == ["Work"]


def test_apple_notes_export_handles_empty_note(tmp_path):
    note = tmp_path / "empty.txt"
    note.write_text("", encoding="utf-8")

    result = AppleNotesExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "empty"
    assert result.units[0].content == ""
