from __future__ import annotations

from graph.rag.query_output_format_requirement import detect_query_output_format_requirements


def test_output_format_requirement_detects_formats_in_order():
    report = detect_query_output_format_requirements("Return JSON, then a checklist and table.")

    assert report["has_format_requirement"] is True
    assert report["formats"] == ["json", "checklist", "table"]
    assert report["primary_format"] == "json"
    assert report["strictness"] == "strict"


def test_output_format_requirement_classifies_preferred_and_incidental_words():
    preferred = detect_query_output_format_requirements("Prefer bullets for the answer.")
    incidental = detect_query_output_format_requirements("Compare the table stakes for this market.")

    assert preferred["formats"] == ["bullet_list"]
    assert preferred["strictness"] == "preferred"
    assert incidental["has_format_requirement"] is False
    assert incidental["formats"] == []
