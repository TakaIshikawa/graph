from __future__ import annotations

from graph.rag.result_confidence_band import estimate_result_confidence_band


def test_result_confidence_band_high_confidence():
    payload = estimate_result_confidence_band(
        [
            {"id": "a", "source_id": "a", "date": "2025-01-01"},
            {"id": "b", "source_id": "b", "date": "2025-01-02"},
            {"id": "c", "source_id": "c", "date": "2025-01-03"},
            {"id": "d", "source_id": "d", "date": "2025-01-04"},
        ],
        now="2025-06-01",
    )

    assert payload["confidence_band"] == "high"
    assert payload["downgrade_reasons"] == []


def test_result_confidence_band_medium_confidence():
    payload = estimate_result_confidence_band(
        [{"id": "a", "source_id": "a", "date": "2025-01-01"}, {"source_id": "a", "date": "2025-01-02"}],
        now="2025-06-01",
    )

    assert payload["confidence_band"] == "medium"
    assert "low source diversity" in payload["downgrade_reasons"]


def test_result_confidence_band_low_confidence_with_contradiction():
    payload = estimate_result_confidence_band([{"source_id": "a", "date": "2020-01-01", "contradiction": True}], now="2025-06-01")

    assert payload["confidence_band"] == "low"
    assert "contradiction flag present" in payload["downgrade_reasons"]


def test_result_confidence_band_empty_result_case():
    payload = estimate_result_confidence_band([])

    assert payload == {
        "confidence_band": "low",
        "confidence_score": 0.0,
        "contributing_factors": [],
        "downgrade_reasons": ["empty result set"],
    }
