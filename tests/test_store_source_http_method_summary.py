from __future__ import annotations

from graph.store.source_http_method_summary import summarize_source_http_methods


def test_source_http_methods_normalizes_and_counts_unsafe_methods():
    summary = summarize_source_http_methods(
        [
            {"id": "a", "metadata": {"method": "get"}},
            {"id": "b", "http_method": "post"},
            {"id": "c", "metadata": {"request_method": "PATCH"}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["method_counts"] == {"GET": 1, "PATCH": 1, "POST": 1}
    assert summary["unsafe_method_count"] == 2
    assert summary["missing_method_count"] == 1


def test_source_http_methods_uses_sample_limit_without_missing_bucket():
    summary = summarize_source_http_methods(
        [{"source_id": "a", "fetch_method": "delete"}, {"source_id": "b"}],
        sample_limit=1,
    )

    assert summary["method_counts"] == {"DELETE": 1}
    assert summary["samples"] == [{"source_id": "a", "field": "fetch_method", "method": "DELETE"}]
