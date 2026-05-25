from __future__ import annotations

from graph.adapters.kindle_clippings_txt import KindleClippingsTxtAdapter


def test_kindle_clippings_txt_ingests_highlights_and_notes(tmp_path):
    export = tmp_path / "My Clippings.txt"
    export.write_text(
        """Deep Work (Cal Newport)
- Your Highlight on page 10 | Location 150-151 | Added on Monday, January 1, 2025 10:30:00 AM

Important highlighted passage.
==========
Deep Work (Cal Newport)
- Your Note on Location 200 | Added on Monday, January 1, 2025 11:00:00 AM

Remember this for planning.
==========
Broken
==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsTxtAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    highlight = result.units[0]
    assert highlight.source_project == "kindle_clippings_txt"
    assert highlight.source_entity_type == "highlight"
    assert highlight.title == "Deep Work"
    assert highlight.metadata["author"] == "Cal Newport"
    assert highlight.metadata["page"] == "10"
    assert highlight.metadata["location"] == "150-151"
    assert "Important highlighted passage." in highlight.content
    note = result.units[1]
    assert note.source_entity_type == "note"
    assert "Remember this for planning." in note.content


def test_kindle_clippings_txt_source_ids_are_stable(tmp_path):
    export = tmp_path / "clippings.txt"
    export.write_text(
        """Book
- Your Highlight at location 7 | Added on 2025-01-01

Stable passage
==========
""",
        encoding="utf-8",
    )

    first = KindleClippingsTxtAdapter(path=str(export)).ingest().units[0]
    second = KindleClippingsTxtAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("kindle_clippings_txt:")
