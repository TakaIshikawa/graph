from __future__ import annotations

from graph.store.source_api_version_summary import summarize_source_api_versions


def test_source_api_version_summary_detects_url_header_and_deprecated_versions():
    summary = summarize_source_api_versions(
        [
            {"id": "a", "url": "https://api.example.test/v1/users"},
            {"id": "b", "metadata": {"url": "https://api.example.test/users?api-version=2024-01-01", "deprecated": True}},
            {"id": "c", "headers": {"X-API-Version": "v2"}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_version"] == 3
    assert summary["version_counts"] == {"2024-01-01": 1, "v1": 1, "v2": 1}
    assert summary["location_counts"] == {"header": 1, "url_path": 1, "url_query": 1}
    assert summary["deprecated_version_count"] == 1
    assert summary["samples"][0] == {"source_id": "a", "version": "v1", "location": "url_path"}
