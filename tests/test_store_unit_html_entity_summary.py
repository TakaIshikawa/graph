from __future__ import annotations

from graph.store.unit_html_entity_summary import summarize_unit_html_entities


def test_html_entity_summary_classifies_counts_frequency_and_samples():
    summary = summarize_unit_html_entities([{"id": "u1", "content": "&amp; &#169; &#x1F600;\n&amp;"}, {"id": "u2", "content": "none"}], sample_limit=2)

    assert summary["total_units"] == 2
    assert summary["units_with_entities"] == 1
    assert summary["named_entity_count"] == 2
    assert summary["decimal_entity_count"] == 1
    assert summary["hex_entity_count"] == 1
    assert summary["entity_frequency"][:2] == [{"entity": "&amp;", "count": 2}, {"entity": "&#169;", "count": 1}]
    assert summary["samples"] == [{"unit_id": "u1", "entity": "&#169;", "line_number": 1}, {"unit_id": "u1", "entity": "&#x1F600;", "line_number": 1}]
