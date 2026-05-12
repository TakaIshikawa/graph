from __future__ import annotations

from graph.rag.query_route import suggest_query_routes


def route_names(report: dict) -> list[str]:
    return [route["route"] for route in report["routes"]]


def test_suggest_query_routes_extracts_tag_prefixed_terms():
    report = suggest_query_routes("find #climate tag:policy adaptation notes")

    assert report["normalized_query"] == "find #climate tag:policy adaptation notes"
    assert report["extracted_tags"] == ["climate", "policy"]
    assert route_names(report)[:2] == ["tag_filter", "semantic"]
    assert report["routes"][0]["confidence"] == 0.9
    assert "climate, policy" in report["routes"][0]["rationale"]


def test_suggest_query_routes_uses_quoted_phrases_for_exact_title():
    report = suggest_query_routes('open "Battery Storage Roadmap" summary')

    assert route_names(report)[0] == "exact_title"
    assert report["routes"][0]["confidence"] == 0.94
    assert "battery storage roadmap" in report["routes"][0]["rationale"]


def test_suggest_query_routes_extracts_source_and_project_hints():
    report = suggest_query_routes("from:readwise project graph source zotero energy")

    assert report["source_hints"] == ["readwise", "graph", "zotero"]
    assert "source_filter" in route_names(report)
    assert "readwise, graph, zotero" in report["rationale"]


def test_suggest_query_routes_extracts_date_words_and_iso_dates():
    report = suggest_query_routes("latest notes since 2026-01-15 before 2026")

    assert report["date_hints"] == ["2026-01-15", "2026", "latest", "since", "before"]
    assert "date_filter" in route_names(report)


def test_suggest_query_routes_detects_contradiction_language():
    report = suggest_query_routes("where do sources conflict or contradict the roadmap")

    assert "contradiction_check" in route_names(report)
    contradiction_route = next(
        route for route in report["routes"] if route["route"] == "contradiction_check"
    )
    assert contradiction_route["confidence"] == 0.88
    assert "conflict" in contradiction_route["rationale"]


def test_suggest_query_routes_empty_queries_do_not_raise():
    assert suggest_query_routes("   ") == {
        "normalized_query": "",
        "routes": [],
        "extracted_tags": [],
        "source_hints": [],
        "date_hints": [],
        "rationale": "Blank or invalid query; no retrieval routes suggested.",
    }
    assert suggest_query_routes(None)["routes"] == []
