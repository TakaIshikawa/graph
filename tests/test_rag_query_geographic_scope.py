from __future__ import annotations

from graph.rag import detect_query_geographic_scope


def test_query_geographic_scope_detects_location_global_and_local_terms():
    assert detect_query_geographic_scope("Find US-only rules")["locations"] == ["United States"]
    assert detect_query_geographic_scope("Compare worldwide availability")["scope_type"] == "global"
    assert detect_query_geographic_scope("Nearby clinics")["local_scope"] is True


def test_query_geographic_scope_returns_none_without_location():
    report = detect_query_geographic_scope("Summarize the evidence")

    assert report["has_geographic_scope"] is False
    assert report["scope_type"] == "none"
