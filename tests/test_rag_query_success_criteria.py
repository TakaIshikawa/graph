from __future__ import annotations

from graph.rag.query_success_criteria import detect_query_success_criteria


def test_extracts_explicit_criteria_and_thresholds():
    report = detect_query_success_criteria("Success means p95 under 200 ms; target at least 99%.")

    assert report["has_explicit_success_criteria"] is True
    assert report["criteria"][0]["text"] == "Success means p95 under 200 ms"
    assert [row["text"] for row in report["numeric_thresholds"]] == ["under 200 ms", "at least 99%"]
    assert report["warnings"] == []


def test_flags_implied_but_not_explicit_criteria():
    report = detect_query_success_criteria("Assess whether the migration is ready.")

    assert report["has_explicit_success_criteria"] is False
    assert report["implied_criteria"] is True
    assert report["warnings"] == ["success_criteria_implied_but_not_explicit"]


def test_neutral_query_has_empty_criteria():
    report = detect_query_success_criteria("What changed in the release notes?")

    assert report["criteria"] == []
    assert report["numeric_thresholds"] == []
    assert report["implied_criteria"] is False
