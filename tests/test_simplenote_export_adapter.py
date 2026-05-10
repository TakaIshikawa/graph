from __future__ import annotations

import json

from graph.adapters.simplenote_export import SimplenoteExportAdapter
from graph.types.enums import SourceProject


def test_simplenote_export_imports():
    """Test that Simplenote export adapter can be instantiated."""
    adapter = SimplenoteExportAdapter()
    assert adapter.name == "simplenote_export"
    assert adapter.entity_types == ["note"]


def test_simplenote_export_parses_json(tmp_path):
    """Test parsing JSON notes."""
    notes_file = tmp_path / "notes.json"
    notes_file.write_text(
        json.dumps([{"id": "123", "content": "My Note\nContent here", "tags": ["work", "personal"]}]),
        encoding="utf-8",
    )

    result = SimplenoteExportAdapter(path=str(notes_file)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SIMPLENOTE_EXPORT
    assert unit.source_entity_type == "note"
    assert unit.title == "My Note"
    assert "work" in unit.tags
    assert "personal" in unit.tags
