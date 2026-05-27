from __future__ import annotations

from graph.store import summarize_unit_markdown_table_alignments


def test_table_alignment_summary_classifies_alignments():
    report = summarize_unit_markdown_table_alignments([{"id": "a", "content": "| A | B | C | D |\n| :-- | --: | :-: | --- |"}])

    assert report["total_tables"] == 1
    assert report["total_columns"] == 4
    assert report["alignment_counts"] == {"center": 1, "left": 1, "right": 1, "unspecified": 1}


def test_table_alignment_summary_counts_malformed_rows():
    report = summarize_unit_markdown_table_alignments([{"content": "| A | B |\n| bad | --- |"}])

    assert report["malformed_delimiter_rows"] == 1
    assert report["total_tables"] == 0


def test_table_alignment_summary_ignores_fenced_code():
    report = summarize_unit_markdown_table_alignments([{"content": "```\n| A | B |\n| -- | -- |\n```"}])

    assert report["total_tables"] == 0
