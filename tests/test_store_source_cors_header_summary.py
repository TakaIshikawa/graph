from __future__ import annotations

from types import SimpleNamespace

from graph.store.source_cors_header_summary import summarize_source_cors_headers


def test_cors_headers_are_detected_from_aliases_and_containers():
    source = SimpleNamespace(
        source_id="obj",
        access_control_allow_origin="https://app.example.test",
        metadata={
            "Access-Control-Allow-Credentials": "true",
            "response_headers": {"ACCESS_CONTROL_ALLOW_METHODS": "GET, POST"},
        },
    )

    summary = summarize_source_cors_headers(
        [
            source,
            {"id": "m1", "headers": {"access_control_allow_headers": "Authorization"}},
            {"id": "m2", "response_headers": {"Access-Control-Allow-Origin": "*"}},
            {"id": "missing"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_cors_headers"] == 3
    assert summary["allow_origin_counts"] == {"*": 1, "https://app.example.test": 1}
    assert summary["credentialed_count"] == 1
    assert summary["wildcard_origin_count"] == 1
    assert summary["missing_header_count"] == 1


def test_cors_samples_are_sorted_and_obey_limit_zero():
    sources = [
        {"id": "z", "Access-Control-Allow-Origin": "https://z.test"},
        {"id": "a", "Access-Control-Allow-Origin": "*"},
    ]

    assert summarize_source_cors_headers(sources, sample_limit=1)["samples"] == [
        {"source_id": "a", "headers": {"access-control-allow-origin": "*"}}
    ]
    assert summarize_source_cors_headers(sources, sample_limit=0)["samples"] == []
