from __future__ import annotations

import json

from graph.adapters.google_keep_export import GoogleKeepExportAdapter
from graph.types.enums import SourceProject


def test_google_keep_export_imports():
    """Test that Google Keep export adapter can be instantiated."""
    adapter = GoogleKeepExportAdapter()
    assert adapter.name == "google_keep_export"
    assert adapter.entity_types == ["note"]


def test_google_keep_export_wraps_google_keep(tmp_path):
    """Test that Google Keep export adapter wraps Google Keep adapter."""
    note = tmp_path / "note.json"
    note.write_text(
        json.dumps({"id": "123", "title": "Test Note", "textContent": "Content"}),
        encoding="utf-8",
    )

    result = GoogleKeepExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_KEEP_EXPORT
    assert unit.title == "Test Note"
