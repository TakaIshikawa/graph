from __future__ import annotations

from graph.rag.query_recency_sensitivity import detect_query_recency_sensitivity


def test_detects_latest_current_today_now():
    result = detect_query_recency_sensitivity("What are the latest rules as of today and now?")

    assert result["requires_recent_evidence"] is True
    assert result["recency_terms"] == ["latest", "today", "now"]


def test_detects_change_oriented_recency():
    result = detect_query_recency_sensitivity("Summarize recent changes in the new policy.")

    assert result["requires_recent_evidence"] is True
    assert "recent_changes" in result["recency_terms"]
    assert "new" in result["recency_terms"]


def test_avoids_now_substrings_and_fixed_past_current():
    assert detect_query_recency_sensitivity("Explain knowledge graphs.")["requires_recent_evidence"] is False
    assert detect_query_recency_sensitivity("What was current in 2020?")["requires_recent_evidence"] is False
