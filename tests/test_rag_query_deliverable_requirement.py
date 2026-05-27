from __future__ import annotations

from graph.rag.query_deliverable_requirement import detect_query_deliverable_requirement


def test_query_deliverable_requirement_detects_multiple_deliverables_in_order():
    report = detect_query_deliverable_requirement("Give me a table, then a checklist and brief.")

    assert report["deliverables"] == ["table", "checklist", "brief"]
    assert report["primary_deliverable"] == "table"
    assert report["confidence"] == 0.85


def test_query_deliverable_requirement_detects_format_specific_cues():
    report = detect_query_deliverable_requirement("Return JSON and CSV exports plus a comparison matrix.")

    assert report["deliverables"] == ["json", "csv", "comparison_matrix"]
    assert report["confidence"] == 0.9


def test_query_deliverable_requirement_deduplicates_duplicate_cues_and_fallback():
    report = detect_query_deliverable_requirement("Use bullets, bullets, and bullet points.")
    fallback = detect_query_deliverable_requirement("What changed?")

    assert report["deliverables"] == ["bullet_list"]
    assert [cue["cue"] for cue in report["matched_cues"]] == ["bullets", "bullet points"]
    assert fallback["deliverables"] == []
    assert fallback["primary_deliverable"] is None
    assert fallback["confidence"] == 0.0
