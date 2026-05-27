from __future__ import annotations

from graph.store.unit_markdown_reference_usage_summary import summarize_unit_markdown_reference_usage


def test_reference_usage_summary_counts_forms_and_samples_unresolved_case_insensitively():
    summary = summarize_unit_markdown_reference_usage([
        {"id": "u1", "content": "[Docs][docs] [Guide][] [Missing]\n[DOCS]: https://example.com\n[Guide]: /guide"},
        {"id": "u2", "content": "No refs"},
    ])

    assert summary == {
        "total_units": 2,
        "units_with_reference_usages": 1,
        "full_usage_count": 1,
        "collapsed_usage_count": 1,
        "shortcut_usage_count": 1,
        "unresolved_label_samples": [{"unit_id": "u1", "label": "Missing", "usage_type": "shortcut", "line_number": 1}],
    }
