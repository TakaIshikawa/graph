from __future__ import annotations

from graph.store import summarize_unit_internal_anchor_targets


def test_summarize_unit_internal_anchor_targets_resolves_headings_and_ids():
    summary = summarize_unit_internal_anchor_targets(
        [
            {"id": "a", "content": "# Heading One\n[ok](#heading-one)\n[bad](#missing)\n## Custom {#custom}\n[x](#custom)\n<div id=\"html-id\"></div>[h](#html-id)"},
            {"id": "b", "content": "# Same\n## Same\n[dup](#same)"},
        ]
    )

    assert summary["resolved_count"] == 4
    assert summary["unresolved_count"] == 1
    assert summary["duplicate_target_count"] == 1
    assert summary["missing_fragment_samples"] == [{"unit_id": "a", "fragment": "missing"}]
