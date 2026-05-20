from __future__ import annotations

from graph.rag.query_temporal_intent import extract_query_temporal_intent


def test_query_temporal_intent_blank_query_returns_neutral_structure():
    assert extract_query_temporal_intent("  \n ") == {
        "has_temporal_intent": False,
        "intent": "none",
        "confidence": 0.0,
        "years": [],
        "iso_dates": [],
        "months": [],
        "ranges": [],
        "recency_cues": [],
        "historical_cues": [],
        "reasons": [],
    }


def test_query_temporal_intent_captures_explicit_years_dates_and_ranges():
    payload = extract_query_temporal_intent("Compare March 2020 to 2022 and 2021-05-04 roadmap updates")

    assert payload["has_temporal_intent"] is True
    assert payload["intent"] == "range"
    assert payload["years"] == ["2020", "2021", "2022"]
    assert payload["iso_dates"] == ["2021-05-04"]
    assert payload["months"] == ["march"]
    assert payload["ranges"] == [{"start": "2020", "end": "2022"}]
    assert payload["confidence"] >= 0.8
    assert payload["reasons"] == [
        "matched ISO date",
        "matched explicit year",
        "matched month name",
        "matched explicit range",
    ]


def test_query_temporal_intent_classifies_recency_separately_from_historical():
    recent = extract_query_temporal_intent("latest source gaps from last week")
    historical = extract_query_temporal_intent("historical archive before 2019")

    assert recent["intent"] == "recency"
    assert recent["recency_cues"] == ["latest", "last week"]
    assert recent["historical_cues"] == []
    assert "matched recency cue" in recent["reasons"]

    assert historical["intent"] == "historical"
    assert historical["historical_cues"] == ["historical", "archive", "before"]
    assert historical["recency_cues"] == []
    assert historical["years"] == ["2019"]


def test_query_temporal_intent_output_is_stable_for_reordered_matches():
    first = extract_query_temporal_intent("June 2024, May 2023, 2024-06-01, latest recent")
    second = extract_query_temporal_intent("recent latest 2024-06-01 May 2023 June 2024")

    assert first["years"] == second["years"] == ["2023", "2024"]
    assert first["iso_dates"] == second["iso_dates"] == ["2024-06-01"]
    assert first["months"] == second["months"] == ["june", "may"]
    assert first["recency_cues"] == second["recency_cues"] == ["latest", "recent"]
