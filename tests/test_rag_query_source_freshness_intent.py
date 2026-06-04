from __future__ import annotations

from graph.rag import detect_query_source_freshness_intent


def test_source_freshness_classifies_current_queries():
    result = detect_query_source_freshness_intent("What are the latest SEC disclosure rules?")

    assert result["intent"] == "current"
    assert result["requires_fresh_sources"] is True
    assert result["suggested_source_date_filter"] == {"mode": "prefer_recent"}
    assert result["confidence"] == 0.86


def test_source_freshness_classifies_historical_and_point_in_time_queries():
    historical = detect_query_source_freshness_intent("Summarize historical guidance in 2022.")
    point = detect_query_source_freshness_intent("What was the policy as of 2024-05-01?")

    assert historical["intent"] == "historical"
    assert historical["requires_fresh_sources"] is False
    assert point["intent"] == "point_in_time"
    assert point["suggested_source_date_filter"] == {"mode": "as_of", "date": "2024-05-01"}


def test_source_freshness_classifies_evergreen_and_unspecified_queries():
    evergreen = detect_query_source_freshness_intent("What is retrieval augmented generation?")
    unspecified = detect_query_source_freshness_intent("Compare embeddings.")

    assert evergreen["intent"] == "evergreen"
    assert evergreen["suggested_source_date_filter"] == {"mode": "no_recency_boost"}
    assert unspecified["intent"] == "unspecified"
    assert unspecified["confidence"] == 0.0
