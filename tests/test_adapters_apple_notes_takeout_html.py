from __future__ import annotations

from graph.adapters import AppleNotesTakeoutHtmlAdapter


def test_apple_notes_takeout_html_adapter_reads_html_and_decodes_entities(tmp_path):
    path = tmp_path / "note.html"
    path.write_text(
        """
        <html><head>
        <title>Title Element</title>
        <meta name="created" content="2026-05-30T00:00:00Z">
        <style>.hidden{}</style><script>alert("x")</script>
        </head><body><h1>Heading</h1><p>Fish &amp; chips</p></body></html>
        """,
        encoding="utf-8",
    )

    adapter = AppleNotesTakeoutHtmlAdapter(str(path))
    result = adapter.ingest()

    assert adapter.name == "apple_notes_takeout_html"
    assert "note" in adapter.entity_types
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Title Element"
    assert "Fish & chips" in unit.content
    assert "alert" not in unit.content
    assert unit.metadata["source_file"] == str(path)


def test_apple_notes_takeout_html_title_fallback_order(tmp_path):
    titled = tmp_path / "a.html"
    heading = tmp_path / "b.html"
    fallback = tmp_path / "file-name.html"
    titled.write_text("<title>From title</title><h1>From heading</h1>", encoding="utf-8")
    heading.write_text("<h1>From heading</h1>", encoding="utf-8")
    fallback.write_text("<p>No title</p>", encoding="utf-8")

    titles = sorted(unit.title for unit in AppleNotesTakeoutHtmlAdapter(str(tmp_path)).ingest().units)

    assert titles == ["From heading", "From title", "file-name"]
