from __future__ import annotations

from graph.store.source_http_method_hint_summary import summarize_source_http_method_hints


def test_http_method_hints_normalize_methods_and_counts():
    summary = summarize_source_http_method_hints(
        [
            {"id": "a", "method": "get"},
            {"id": "b", "metadata": {"http_method": "post"}},
            {"id": "c", "request": {"request_method": "PATCH"}},
            {"id": "d", "options": {"method": "delete"}},
            {"id": "e", "method": "HEAD"},
            {"id": "f"},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_method_hint"] == 5
    assert summary["method_counts"] == {"DELETE": 1, "GET": 1, "HEAD": 1, "PATCH": 1, "POST": 1}
    assert summary["non_get_count"] == 4
    assert summary["unsafe_method_count"] == 3
    assert summary["missing_method_hint_count"] == 1


def test_http_method_samples_are_deterministic_and_limited():
    summary = summarize_source_http_method_hints(
        [{"id": "z", "method": "PUT"}, {"id": "a", "metadata": {"request": {"method": "options"}}}],
        sample_limit=1,
    )

    assert summary["samples"] == [{"source_id": "a", "method": "OPTIONS", "field": "metadata.request.method"}]
    assert summarize_source_http_method_hints([{"id": "a", "method": "GET"}], sample_limit=0)["samples"] == []
