from __future__ import annotations

from graph.rag.query_jurisdiction_requirement import detect_query_jurisdiction_requirement


def test_marks_high_stakes_query_without_jurisdiction_as_missing():
    report = detect_query_jurisdiction_requirement("Can my employer require overtime without extra pay?")

    assert report["requires_jurisdiction"] is True
    assert report["missing_jurisdiction"] is True
    assert "employment" in report["domains"]


def test_recognizes_country_us_state_eu_uk_and_canada():
    report = detect_query_jurisdiction_requirement("Compare tax rules in California, the EU, UK, and Canada.")

    assert report["missing_jurisdiction"] is False
    assert {"California", "EU", "UK", "Canada"} <= set(report["detected_jurisdictions"])
    assert "tax" in report["domains"]


def test_generic_location_specific_cue_requires_but_is_not_missing():
    report = detect_query_jurisdiction_requirement("What local law applies where I live?")

    assert report["requires_jurisdiction"] is True
    assert report["missing_jurisdiction"] is False
    assert "generic_location_specific_cue" in report["reasons"]
