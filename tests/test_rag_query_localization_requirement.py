from __future__ import annotations

from graph.rag.query_localization_requirement import detect_query_localization_requirements


def test_detect_query_localization_requirements_mixed_categories():
    rows = detect_query_localization_requirements(
        "Need multilingual support, locale rules, currency conversion, time zones, date format, and metric units."
    )

    assert [row["category"] for row in rows] == [
        "language",
        "locale",
        "currency",
        "timezone",
        "regional_format",
        "units",
    ]


def test_detect_query_localization_requirements_deduplicates_repeated_category():
    rows = detect_query_localization_requirements("Translate the language copy and support multiple languages.")

    assert rows == [{"matched_text": "translate", "category": "language", "severity": "high", "span": [0, 9]}]


def test_detect_query_localization_requirements_normalizes_whitespace():
    rows = detect_query_localization_requirements("Show local\n time and regional\t formatting.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("timezone", "local time"),
        ("regional_format", "regional formatting"),
    ]
