from __future__ import annotations

from graph.rag.query_recency_requirement import detect_query_recency_requirement


def test_query_recency_requirement_no_recency_query_is_neutral():
    payload = detect_query_recency_requirement("Explain the history of database indexes")

    assert payload == {
        "query": "Explain the history of database indexes",
        "requires_recency": False,
        "recency_window": "none",
        "matched_terms": [],
        "retrieval_hint": "no recency preference detected",
    }


def test_query_recency_requirement_detects_vague_recency_terms():
    payload = detect_query_recency_requirement("What are the latest current Kubernetes limits?")

    assert payload["requires_recency"] is True
    assert payload["recency_window"] == "current"
    assert payload["matched_terms"] == ["current", "latest"]
    assert payload["retrieval_hint"] == "prefer the most recently updated authoritative sources"


def test_query_recency_requirement_detects_explicit_windows():
    payload = detect_query_recency_requirement("Summarize incidents from last 7 days and this month")

    assert payload["requires_recency"] is True
    assert payload["recency_window"] == "P7D"
    assert payload["matched_terms"] == ["last 7 days", "this month"]
    assert "P7D" in payload["retrieval_hint"]


def test_query_recency_requirement_detects_recently_as_vague():
    payload = detect_query_recency_requirement("Has the policy changed recently?")

    assert payload["recency_window"] == "recent"
    assert payload["matched_terms"] == ["recently"]
