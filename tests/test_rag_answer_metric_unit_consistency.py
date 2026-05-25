from __future__ import annotations

from graph.rag.answer_metric_unit_consistency import audit_answer_metric_unit_consistency


def test_extracts_numeric_unit_mentions():
    report = audit_answer_metric_unit_consistency("Latency was 100 ms and cost was $20 monthly.")

    assert [row["text"] for row in report["unit_mentions"]] == ["100 ms", "$20"]
    assert report["warnings"] == []


def test_flags_inconsistent_units_without_conversion():
    report = audit_answer_metric_unit_consistency("Latency is 100 ms in one source and 2 seconds in another. Rates are 10 monthly and 120 annual.")

    assert {"family": "time", "units": ["ms", "seconds"]} in report["inconsistent_unit_families"]
    assert {"family": "rate_period", "units": ["annual", "monthly"]} in report["inconsistent_unit_families"]
    assert "inconsistent_time_units_without_conversion" in report["warnings"]


def test_allows_mixed_units_with_conversion_language():
    report = audit_answer_metric_unit_consistency("Latency is 100 ms, equivalent to 0.1 seconds after conversion.")

    assert report["has_conversion_language"] is True
    assert report["inconsistent_unit_families"] == []
