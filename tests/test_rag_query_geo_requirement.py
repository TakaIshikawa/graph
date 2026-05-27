from __future__ import annotations

from graph.rag.query_geo_requirement import detect_query_geo_requirements


def test_query_geo_requirement_detects_local_phrasing():
    report = detect_query_geo_requirements("Find local pediatric clinics near me")

    assert report["has_geo_requirement"] is True
    assert report["scope_type"] == "local"
    assert report["geo_terms"] == ["local", "near me"]


def test_query_geo_requirement_detects_country_and_region_terms():
    report = detect_query_geo_requirements("Compare privacy rules in the EU and United States")

    assert report["has_geo_requirement"] is True
    assert report["scope_type"] == "country"
    assert report["geo_terms"] == ["United States", "EU"]


def test_query_geo_requirement_detects_city_and_global_scope():
    city = detect_query_geo_requirements("Best EV rebates in New York")
    global_report = detect_query_geo_requirements("Show global cloud adoption statistics")

    assert city["scope_type"] == "city"
    assert city["geo_terms"] == ["New York"]
    assert global_report["scope_type"] == "global"
    assert global_report["geo_terms"] == ["global"]


def test_query_geo_requirement_ignores_empty_or_non_geo_queries():
    report = detect_query_geo_requirements("Explain vector databases")

    assert report["has_geo_requirement"] is False
    assert report["geo_terms"] == []
    assert report["scope_type"] == "none"
