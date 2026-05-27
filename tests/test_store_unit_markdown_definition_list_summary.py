from __future__ import annotations

from graph.store import summarize_unit_markdown_definition_lists


def test_definition_list_summary_counts_blocks_and_ignores_paragraph_colons():
    report = summarize_unit_markdown_definition_lists([{"id": "u", "content": "Term\n: one\n: two\n\nLoose\n\n: spaced\nhttp://x: no"}])

    assert report["definition_list_blocks"] == 2
    assert report["term_count"] == 2
    assert report["definition_count"] == 3
    assert report["multi_definition_term_count"] == 1
    assert report["loose_spacing_variant_count"] == 1


def test_definition_list_summary_groups_stacked_terms():
    report = summarize_unit_markdown_definition_lists([{"id": "u", "content": "Term A\nTerm B\n: shared definition"}])

    assert report["definition_list_blocks"] == 1
    assert report["term_count"] == 2
    assert report["definition_count"] == 2
