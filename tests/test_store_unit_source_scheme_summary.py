from __future__ import annotations

from graph.store import summarize_unit_source_schemes


def test_source_scheme_summary_classifies_sources_and_examples():
    summary = summarize_unit_source_schemes(
        [
            {"id": "https", "metadata": {"source_url": "https://example.com/a"}},
            {"id": "file", "metadata": {"source_path": "file:///tmp/a.md"}},
            {"id": "relative", "metadata": {"source_path": "notes/a.md"}},
            {"id": "missing", "metadata": {}},
            {"id": "malformed", "metadata": {"url": "https:/missing-host"}},
            {"id": "custom", "metadata": {"url": "bear://x/y"}},
        ]
    )

    rows = {row["scheme"]: row for row in summary["rows"]}
    assert rows["https"]["unit_count"] == 1
    assert rows["file"]["example_sources"] == ["file:///tmp/a.md"]
    assert rows["relative"]["example_unit_ids"] == ["relative"]
    assert rows["missing"]["unit_count"] == 1
    assert rows["malformed"]["example_unit_ids"] == ["malformed"]
    assert rows["bear"]["unit_count"] == 1
