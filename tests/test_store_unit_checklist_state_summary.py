from __future__ import annotations

from graph.store import summarize_unit_checklist_states


def test_checklist_state_summary_counts_markers_states_and_nested_items():
    report = summarize_unit_checklist_states([{"id": "u", "content": "- [ ] Open\n  - [x] Done\n- [?] Custom\n```\n- [ ] Ignore\n```"}])

    assert report["total_items"] == 3
    assert report["nested_item_count"] == 1
    assert {row["state"]: row["count"] for row in report["normalized_state_counts"]} == {"custom": 1, "done": 1, "open": 1}
    assert report["examples_by_state"]["done"][0]["item_text"] == "Done"
