from __future__ import annotations

from graph.store import summarize_source_openapi_hints


def test_source_openapi_hints_count_documentation_cues_and_spec_urls():
    summary = summarize_source_openapi_hints(
        [
            {"id": "s1", "url": "https://api.example.test/openapi.json"},
            {"id": "s2", "metadata": {"docs": "Swagger UI at https://api.example.test/swagger.json"}},
            {"id": "s3", "description": "API docs rendered with Redoc and Scalar."},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_openapi_hints"] == 3
    assert summary["hint_counts"] == {
        "api_docs": 1,
        "openapi": 1,
        "redoc": 1,
        "scalar": 1,
        "swagger": 1,
    }
    assert "https://api.example.test/openapi.json" in summary["likely_spec_urls"]
    assert "https://api.example.test/swagger.json" in summary["likely_spec_urls"]


def test_source_openapi_hints_include_capped_samples_with_category():
    summary = summarize_source_openapi_hints(
        [
            {"id": "s1", "url": "https://api.example.test/openapi.json"},
            {"id": "s2", "url": "https://api.example.test/api-docs"},
        ],
        sample_limit=1,
    )

    assert summary["samples"] == [
        {
            "source_id": "s1",
            "field": "url",
            "category": "openapi",
            "matched_text": "https://api.example.test/openapi.json",
        }
    ]


def test_source_openapi_hints_ignore_unrelated_sources():
    assert summarize_source_openapi_hints([{"id": "s1", "url": "https://example.test/help"}]) == {
        "total_sources": 1,
        "sources_with_openapi_hints": 0,
        "hint_counts": {},
        "likely_spec_urls": [],
        "samples": [],
    }
