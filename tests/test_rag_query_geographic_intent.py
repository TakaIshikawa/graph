from __future__ import annotations

from graph.rag.query_geographic_intent import detect_query_geographic_intent


def test_geographic_intent_blank_query_returns_neutral_structure():
    payload = detect_query_geographic_intent("  \n ")

    assert payload == {
        "normalized_query": "",
        "local": False,
        "global_remote": False,
        "explicit_location": False,
        "location_comparison": False,
        "location_like_terms": [],
        "reasons": {
            "local": [],
            "global_remote": [],
            "explicit_location": [],
            "location_comparison": [],
        },
    }


def test_geographic_intent_extracts_city_and_near_cues():
    payload = detect_query_geographic_intent("Find local vendors near Boston and in Tokyo")

    assert payload["local"] is True
    assert payload["explicit_location"] is True
    assert payload["location_like_terms"] == ["Boston", "Tokyo"]
    assert payload["reasons"]["local"] == ["near", "local"]
    assert payload["reasons"]["explicit_location"] == ["location phrase"]


def test_geographic_intent_extracts_regions_and_global_remote_cues():
    payload = detect_query_geographic_intent("Remote roles available globally for US and EU teams")

    assert payload["global_remote"] is True
    assert payload["explicit_location"] is True
    assert payload["location_like_terms"] == ["EU", "US"]
    assert payload["reasons"]["global_remote"] == ["global", "remote"]


def test_geographic_intent_detects_location_comparisons():
    payload = detect_query_geographic_intent("Compare retention in Tokyo vs near Boston by region")

    assert payload["location_comparison"] is True
    assert payload["local"] is True
    assert payload["explicit_location"] is True
    assert payload["location_like_terms"] == ["Boston", "Tokyo"]
    assert payload["reasons"]["location_comparison"] == ["versus", "compare", "by region"]
