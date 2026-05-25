from __future__ import annotations

import pytest

from graph.rag.query_locale_requirement import detect_query_locale_requirement


def test_locale_requirement_detects_explicit_locale_phrases():
    result = detect_query_locale_requirement(
        "Format this for British English and country-specific UK locale conventions."
    )

    assert result["requires_locale_awareness"] is True
    assert result["locale_cues"] == ["language_variant", "country_specific", "locale"]
    assert result["format_cues"] == []
    assert result["normalization_recommendations"] == ["resolve_requested_locale_before_retrieval_and_formatting"]
    assert result["confidence"] == 0.85


def test_locale_requirement_detects_implicit_date_address_and_number_formats():
    result = detect_query_locale_requirement(
        "Parse DD/MM/YYYY dates, postcode before city address format, comma decimal 1,23, and metric units."
    )

    assert result["locale_cues"] == []
    assert result["format_cues"] == [
        "date_format",
        "address_format",
        "postal_code",
        "decimal_separator",
        "measurement_convention",
    ]
    assert result["normalization_recommendations"] == [
        "normalize_ambiguous_dates_to_iso_8601",
        "use_country_specific_address_and_postal_code_parsing",
        "preserve_locale_units_and_number_separators_in_answer",
    ]


def test_locale_requirement_no_cues_is_false():
    result = detect_query_locale_requirement("Explain semantic search.")

    assert result["requires_locale_awareness"] is False
    assert result["locale_cues"] == []
    assert result["format_cues"] == []
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", " ", None])
def test_locale_requirement_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        detect_query_locale_requirement(query)  # type: ignore[arg-type]
