from __future__ import annotations

from graph.store import summarize_unit_heading_hierarchy


def test_summarize_unit_heading_hierarchy_counts_levels_and_skips():
    summary = summarize_unit_heading_hierarchy([{"id": "a", "title": "A", "content": "# One\n### Three"}, {"id": "b", "content": "plain"}])

    assert summary["heading_counts_by_level"] == {"1": 1, "3": 1}
    assert summary["max_depth"] == 3
    assert summary["units_with_skipped_levels"] == 1
    assert summary["skipped_level_samples"] == [{"unit_id": "a", "title": "A", "line_number": 2, "level": 3}]
