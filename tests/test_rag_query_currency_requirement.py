from __future__ import annotations

import pytest

from graph.rag.query_currency_requirement import detect_query_currency_requirement


def test_currency_requirement_detects_symbols_and_codes_without_duplicates():
    result = detect_query_currency_requirement("Compare $ revenue in USD, EUR, and ¥ costs.")

    assert result["requires_currency_normalization"] is True
    assert result["explicit_currencies"] == ["USD", "EUR", "JPY"]
    assert result["conversion_cues"] == []
    assert result["confidence"] == 0.9


def test_currency_requirement_separates_conversion_and_adjustment_cues():
    result = detect_query_currency_requirement(
        "Convert local currency using exchange rate and compare inflation-adjusted real terms, not nominal."
    )

    assert result["conversion_cues"] == ["exchange_rate", "local_currency", "convert_currency"]
    assert result["adjustment_cues"] == ["inflation_adjusted", "nominal", "real_terms"]
    assert result["recommendations"] == [
        "normalize_monetary_values_to_requested_or_reference_currency",
        "separate_nominal_values_from_inflation_adjusted_real_terms",
    ]


def test_currency_requirement_no_cues_is_false():
    result = detect_query_currency_requirement("Compare adoption rates.")

    assert result["requires_currency_normalization"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", None])
def test_currency_requirement_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_currency_requirement(query)  # type: ignore[arg-type]
