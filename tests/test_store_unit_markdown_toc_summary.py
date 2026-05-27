from __future__ import annotations

from graph.store.unit_markdown_toc_summary import summarize_unit_markdown_toc


def test_toc_summary_counts_depth_unresolved_and_duplicate_fragments():
    summary = summarize_unit_markdown_toc([
        {"id": "u1", "content": "- [Intro](#intro)\n  - [Missing](#missing)\n  - [Missing Again](#missing)\n# Intro\n## Other {#custom-id}"},
        {"id": "u2", "content": "No TOC"},
    ])

    assert summary == {
        "total_units": 2,
        "units_with_toc_entries": 1,
        "toc_entry_count": 3,
        "max_toc_depth": 2,
        "unresolved_toc_fragment_samples": [
            {"unit_id": "u1", "fragment": "missing", "line_number": 2},
            {"unit_id": "u1", "fragment": "missing", "line_number": 3},
        ],
        "duplicate_toc_fragment_samples": [
            {"unit_id": "u1", "fragment": "missing", "line_number": 2},
            {"unit_id": "u1", "fragment": "missing", "line_number": 3},
        ],
    }
