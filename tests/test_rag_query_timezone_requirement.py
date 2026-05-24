from __future__ import annotations

import pytest

from graph.rag.query_timezone_requirement import detect_query_timezone_requirement


def test_timezone_requirement_flags_relative_terms_and_location_hints():
    result = detect_query_timezone_requirement("What changed today before market open in New York local time?")

    assert result["requires_timezone_awareness"] is True
    assert result["relative_time_terms"] == ["today", "market open", "local time"]
    assert result["explicit_timezones"] == []
    assert result["location_hints"] == ["New York"]
    assert result["confidence"] == 0.8
    assert result["normalization_recommendations"] == [
        "resolve_relative_terms_against_user_or_query_timezone",
        "map_location_hints_to_iana_timezones_before_date_math",
    ]


def test_timezone_requirement_extracts_abbreviations_and_offsets_separately():
    result = detect_query_timezone_requirement("Convert yesterday 9:00 PST to UTC+09:00 and GMT-5.")

    assert result["relative_time_terms"] == ["yesterday"]
    assert result["explicit_timezones"] == ["GMT-5", "PST", "UTC+09:00"]
    assert result["location_hints"] == []
    assert result["normalization_recommendations"] == ["normalize_times_to_utc_and_preserve_source_timezone"]
    assert result["confidence"] == 0.9


def test_timezone_requirement_no_match_is_false():
    result = detect_query_timezone_requirement("Explain vector search.")

    assert result["requires_timezone_awareness"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", "  ", None])
def test_timezone_requirement_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        detect_query_timezone_requirement(query)  # type: ignore[arg-type]
