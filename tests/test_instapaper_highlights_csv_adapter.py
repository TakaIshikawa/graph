from __future__ import annotations

from graph.adapters.instapaper_highlights_csv import InstapaperHighlightsCsvAdapter
from graph.adapters.registry import get_adapter


def test_instapaper_highlights_csv_ingests_highlight_rows(tmp_path):
    export = tmp_path / "highlights.csv"
    export.write_text("URL,Title,Highlight,Note,Created\nhttps://example.com,Article,Important quote,My note,2024-01-01T00:00:00Z\n", encoding="utf-8")

    unit = InstapaperHighlightsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id.startswith("instapaper_highlights_csv:")
    assert unit.title == "Article"
    assert unit.metadata["url"] == "https://example.com"
    assert unit.metadata["highlight"] == "Important quote"
    assert unit.metadata["note"] == "My note"
    assert "Important quote" in unit.content


def test_instapaper_highlights_csv_skips_blank_highlights_and_registers(tmp_path):
    export = tmp_path / "highlights.csv"
    export.write_text("URL,Title,Highlight\nhttps://example.com,Article,\n", encoding="utf-8")

    assert InstapaperHighlightsCsvAdapter(path=str(export)).ingest().units == []
    assert isinstance(get_adapter("instapaper_highlights_csv"), InstapaperHighlightsCsvAdapter)
