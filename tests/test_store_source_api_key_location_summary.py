from __future__ import annotations

from graph.store.source_api_key_location_summary import summarize_source_api_key_locations


def test_source_api_key_locations_classify_common_locations():
    summary = summarize_source_api_key_locations(
        [
            {"id": "s1", "headers": {"x-api-key": "required"}},
            {"id": "s2", "authorization": "Bearer token"},
            {"id": "s3", "metadata": {"body": "request body includes api_key"}},
            {"id": "s4", "metadata": {"auth": "API key from ENV var NEWS_API_KEY"}},
        ]
    )

    assert summary["api_key_source_count"] == 4
    assert summary["location_counts"] == {"bearer": 1, "body": 1, "env": 1, "header": 2}
    assert summary["insecure_query_count"] == 0
    assert summary["samples"][0]["source_id"] == "s1"
    assert {"source_id", "field", "location"} <= set(summary["samples"][0])


def test_source_api_key_locations_count_insecure_query_hints():
    summary = summarize_source_api_key_locations(
        [
            {"id": "s1", "url": "https://api.example.test/search?api_key=abc"},
            {"id": "s2", "metadata": {"query": "pass token query parameter"}},
        ]
    )

    assert summary["api_key_source_count"] == 2
    assert summary["location_counts"] == {"query": 2}
    assert summary["insecure_query_count"] == 2


def test_source_api_key_locations_ignore_non_auth_sources():
    assert summarize_source_api_key_locations([{"id": "s1", "url": "https://example.test/public"}]) == {
        "total_sources": 1,
        "api_key_source_count": 0,
        "location_counts": {},
        "insecure_query_count": 0,
        "samples": [],
    }
