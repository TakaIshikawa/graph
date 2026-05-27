from __future__ import annotations

from graph.adapters.google_docs_takeout_html import GoogleDocsTakeoutHtmlAdapter
from graph.adapters.registry import get_adapter


def test_google_docs_takeout_html_recurses_and_extracts_readable_metadata(tmp_path):
    nested = tmp_path / "docs"
    nested.mkdir()
    (nested / "doc.html").write_text("<html><head><title>Doc</title><style>.x{}</style></head><body><h1>Heading</h1><p>Hello <a href='https://example.test'>link</a></p><script>noise</script></body></html>", encoding="utf-8")

    unit = GoogleDocsTakeoutHtmlAdapter(str(tmp_path)).ingest().units[0]

    assert unit.title == "Doc"
    assert "Hello" in unit.content
    assert "noise" not in unit.content
    assert unit.metadata["headings"] == ["Heading"]
    assert unit.metadata["links"] == [{"text": "link", "url": "https://example.test"}]
    assert isinstance(get_adapter("google_docs_takeout_html"), GoogleDocsTakeoutHtmlAdapter)
