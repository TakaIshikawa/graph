from __future__ import annotations

from graph.store import summarize_unit_language_coverage


def test_language_coverage_normalizes_common_language_keys():
    summary = summarize_unit_language_coverage(
        [
            {"id": "a", "metadata": {"language": "EN"}},
            {"id": "b", "metadata": {"locale": "en_US"}},
            {"id": "c", "metadata": {"detected_language": "ja"}},
            {"id": "d", "metadata": {"language": ""}},
            {"id": "e"},
        ]
    )

    assert summary == {
        "total_units": 5,
        "units_with_language": 3,
        "units_without_language": 2,
        "coverage_ratio": "0.60",
        "language_counts": {"en": 2, "ja": 1},
    }


def test_language_coverage_handles_empty_units():
    assert summarize_unit_language_coverage([])["coverage_ratio"] == "0.00"
