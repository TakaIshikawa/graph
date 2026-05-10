from __future__ import annotations

from graph.adapters.bear_export import BearExportAdapter
from graph.types.enums import SourceProject


def test_bear_export_imports():
    """Test that Bear export adapter can be instantiated."""
    adapter = BearExportAdapter()
    assert adapter.name == "bear_export"
    assert adapter.entity_types == ["note"]


def test_bear_export_parses_markdown_with_tags(tmp_path):
    """Test parsing markdown with Bear-style tags."""
    note = tmp_path / "note.md"
    note.write_text("# My Note\n\n#work/project #personal\n\nContent here.", encoding="utf-8")

    result = BearExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.BEAR_EXPORT
    assert unit.source_entity_type == "note"
    assert unit.title == "My Note"
    assert "work/project" in unit.tags
    assert "personal" in unit.tags
