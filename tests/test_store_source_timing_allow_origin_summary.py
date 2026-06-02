from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_source_timing_allow_origins
from graph.store.source_timing_allow_origin_summary import summarize_source_timing_allow_origins as direct_import


def test_timing_allow_origin_summary_counts_aliases_and_nested_headers():
    source = SimpleNamespace(
        source_id="obj",
        timing_allow_origin="https://direct.example.test, *",
        metadata={
            "response_headers": {
                "TIMING_ALLOW_ORIGIN": [
                    "https://meta.example.test",
                    ("https://nested.example.test, https://direct.example.test",),
                ]
            }
        },
    )

    summary = summarize_source_timing_allow_origins(
        [
            source,
            {"id": "headers", "headers": {"Timing-Allow-Origin": "https://headers.example.test"}},
            {"id": "response", "response_headers": {"timing_allow_origin": "*"}},
            {"id": "empty", "metadata": {"Timing-Allow-Origin": ""}},
            {"id": "missing"},
        ]
    )

    assert direct_import is summarize_source_timing_allow_origins
    assert summary["total_sources"] == 5
    assert summary["sources_with_timing_allow_origin"] == 3
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["wildcard_origin_count"] == 2
    assert summary["explicit_origin_count"] == 4
    assert summary["multi_origin_source_count"] == 1
    assert summary["origin_counts"] == {
        "*": 2,
        "https://direct.example.test": 1,
        "https://headers.example.test": 1,
        "https://meta.example.test": 1,
        "https://nested.example.test": 1,
    }


def test_timing_allow_origin_samples_are_sorted_and_obey_limit_zero():
    sources = [
        {"id": "z", "Timing-Allow-Origin": "https://z.example.test, https://a.example.test"},
        {"id": "a", "metadata": {"headers": {"timing-allow-origin": "*"}}},
    ]

    assert summarize_source_timing_allow_origins(sources, sample_limit=1)["samples"] == [
        {"source_id": "a", "origins": ["*"]}
    ]
    assert summarize_source_timing_allow_origins(sources, sample_limit=0)["samples"] == []
