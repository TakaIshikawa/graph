from __future__ import annotations

from graph.rag.query_stakes import classify_query_stakes


def test_query_stakes_maps_high_stakes_domains():
    report = classify_query_stakes("Can this drug dosage affect my tax filing and SSN privacy?")

    assert report["stakes"] == "high"
    assert [row["domain"] for row in report["domains"]] == ["medical", "financial", "privacy"]
    assert "prefer_primary_sources" in report["safeguards"]


def test_query_stakes_keeps_neutral_queries_low():
    report = classify_query_stakes("What is the history of public libraries?")

    assert report == {"stakes": "low", "domains": [], "safeguards": []}
