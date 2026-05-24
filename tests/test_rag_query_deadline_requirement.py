from __future__ import annotations

from graph.rag.query_deadline_requirement import detect_query_deadline_requirement


def test_detects_exact_relative_and_urgency_deadlines():
    report = detect_query_deadline_requirement("Need this ASAP, by Friday, and before launch on 2026-06-01.")

    assert report["has_deadline_requirement"] is True
    assert report["urgency_level"] == "high"
    assert [row["type"] for row in report["matched_cues"]] == ["urgency", "relative_deadline", "relative_deadline", "exact_date"]
    assert [row["text"] for row in report["extracted_deadlines"]] == ["ASAP", "Friday", "launch", "2026-06-01"]
    assert "ambiguous_relative_deadline:friday" in report["warnings"]


def test_detects_within_window_and_empty_input_is_neutral():
    report = detect_query_deadline_requirement("Can we finish within 30 days?")

    assert report["urgency_level"] == "medium"
    assert report["extracted_deadlines"][0]["text"] == "30 days"
    assert detect_query_deadline_requirement("") == {
        "has_deadline_requirement": False,
        "urgency_level": "none",
        "matched_cues": [],
        "extracted_deadlines": [],
        "warnings": [],
    }


def test_ordinary_historical_dates_do_not_create_deadline_requirement():
    report = detect_query_deadline_requirement("Summarize incidents in 2024 and after 2025.")

    assert report["has_deadline_requirement"] is False
    assert report["matched_cues"] == []
