from __future__ import annotations

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
