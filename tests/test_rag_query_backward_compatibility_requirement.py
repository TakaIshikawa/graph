from __future__ import annotations

from graph.rag.query_backward_compatibility_requirement import (
    detect_query_backward_compatibility_requirements,
)


def test_detect_query_backward_compatibility_requirements_category_coverage():
    rows = detect_query_backward_compatibility_requirements(
        "Keep it backwards-compatible, include legacy support, avoid breaking changes, publish a compatibility matrix, and test older clients."
    )

    assert [row["category"] for row in rows] == [
        "backward_compatible",
        "legacy_support",
        "breaking_change",
        "compatibility_matrix",
        "older_clients",
    ]


def test_detect_query_backward_compatibility_requirements_severity_and_sorting():
    rows = detect_query_backward_compatibility_requirements("Flag breaking changes before legacy clients and older versions.")

    assert [(row["category"], row["severity"], row["matched_text"]) for row in rows] == [
        ("breaking_change", "high", "breaking changes"),
        ("legacy_support", "medium", "legacy clients"),
        ("older_clients", "medium", "older versions"),
    ]


def test_detect_query_backward_compatibility_requirements_no_match():
    assert detect_query_backward_compatibility_requirements("Summarize the newest API features.") == []
