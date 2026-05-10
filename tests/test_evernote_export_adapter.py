from __future__ import annotations

from graph.adapters.evernote_export import EvernoteExportAdapter
from graph.types.enums import SourceProject


def test_evernote_export_imports():
    """Test that Evernote export adapter can be instantiated."""
    adapter = EvernoteExportAdapter()
    assert adapter.name == "evernote_export"
    assert adapter.entity_types == ["note"]


def test_evernote_export_wraps_enex(tmp_path):
    """Test that Evernote export adapter wraps ENEX adapter."""
    enex = tmp_path / "notes.enex"
    enex.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE en-export SYSTEM "http://xml.evernote.com/pub/evernote-export4.dtd">
<en-export>
  <note>
    <title>Test Note</title>
    <content><![CDATA[<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">
<en-note>Test content</en-note>]]></content>
    <created>20240101T120000Z</created>
  </note>
</en-export>""",
        encoding="utf-8",
    )

    result = EvernoteExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.EVERNOTE_EXPORT
    assert unit.title == "Test Note"
