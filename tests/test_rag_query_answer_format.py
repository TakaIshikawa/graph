from __future__ import annotations

from graph.rag.query_answer_format import plan_query_answer_format


def test_query_answer_format_maps_comparison_and_timeline_cues():
    comparison = plan_query_answer_format("Compare Alpha vs Beta in a table with citations")
    timeline = plan_query_answer_format("Give me a chronological timeline of releases")

    assert comparison["formats"] == ["table", "comparison", "citations"]
    assert comparison["sections"] == ["criteria", "options", "tradeoffs", "recommendation"]
    assert timeline["formats"] == ["timeline"]
    assert timeline["ordering_hints"] == ["chronological"]


def test_query_answer_format_warns_for_json_plus_prose_table():
    report = plan_query_answer_format("Return JSON and a table checklist")

    assert report["formats"] == ["json", "table", "checklist"]
    assert report["warnings"] == ["conflicting_structured_format_requests"]
