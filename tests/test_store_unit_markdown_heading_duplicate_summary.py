from __future__ import annotations

from graph.store import summarize_unit_markdown_heading_duplicates


def test_summarize_unit_markdown_heading_duplicates_detects_normalized_duplicates():
    summary = summarize_unit_markdown_heading_duplicates([{"id": "a", "content": "# Intro\n##  intro \nText # not heading"}])
    assert summary["units_with_duplicate_headings"] == 1
    assert summary["units"][0]["duplicates"] == [{"heading": "intro", "count": 2}]


def test_summarize_unit_markdown_heading_duplicates_ignores_unique():
    assert summarize_unit_markdown_heading_duplicates([{"content": "# One\n## Two"}])["duplicate_heading_count"] == 0
