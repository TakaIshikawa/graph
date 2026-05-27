from __future__ import annotations

from graph.store.unit_footnote_orphan_summary import summarize_unit_footnote_orphans


def test_summarize_unit_footnote_orphans_counts_unresolved_and_unused():
    summary = summarize_unit_footnote_orphans(
        [
            {"id": "ok", "content": "Use[^a]\n\n[^a]: Done"},
            {"id": "bad", "content": "Use[^missing]\n\n[^unused]: Nope\n```md\n[^ignored]: x\nUse[^ignored]\n```"},
        ]
    )

    assert summary == {
        "total_units": 2,
        "unresolved_references": 1,
        "unused_definitions": 1,
        "affected_units": 1,
        "examples": [
            {"unit_id": "bad", "line": 1, "label": "missing", "orphan_type": "unresolved_reference"},
            {"unit_id": "bad", "line": 3, "label": "unused", "orphan_type": "unused_definition"},
        ],
    }
