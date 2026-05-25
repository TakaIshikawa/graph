from __future__ import annotations

from graph.rag.result_source_canonicalization import analyze_result_source_canonicalization


def test_groups_duplicate_sources_from_url_variants():
    report = analyze_result_source_canonicalization(
        [
            {"id": "a", "source_url": "https://Example.com/reports/alpha/?utm_source=news#summary"},
            {"id": "b", "metadata": {"source_url": "https://example.com/reports/alpha"}},
            {"id": "c", "source_url": "https://example.com/reports/beta/"},
            {"id": "d", "source_url": "https://other.example.com/reports/alpha/"},
        ]
    )

    assert report["canonical_groups"] == [
        {
            "canonical_source": "https://example.com/reports/alpha",
            "result_ids": ["a", "b"],
            "source_values": [
                "https://Example.com/reports/alpha/?utm_source=news#summary",
                "https://example.com/reports/alpha",
            ],
        }
    ]
    assert report["duplicate_result_ids"] == ["b"]
    assert report["unique_source_count"] == 3
    assert report["recommendations"] == [
        "collapse_duplicate_canonical_sources_before_answering",
        "replace_duplicate_results_with_distinct_sources_when_source_diversity_matters",
    ]


def test_canonicalization_preserves_distinct_paths_and_non_tracking_query_params():
    report = analyze_result_source_canonicalization(
        [
            {"id": "a", "source_url": "https://example.com/report?version=1&utm_medium=email"},
            {"id": "b", "source_url": "https://example.com/report?version=2"},
            {"id": "c", "source_url": "https://example.com/report#section"},
        ]
    )

    assert report["canonical_groups"] == []
    assert report["duplicate_result_ids"] == []
    assert report["unique_source_count"] == 3


def test_uses_domain_fields_when_url_is_absent():
    report = analyze_result_source_canonicalization(
        [
            {"id": "a", "metadata": {"domain": "Example.com"}},
            {"id": "b", "domain": "https://www.example.com/"},
        ]
    )

    assert report["canonical_groups"] == [
        {
            "canonical_source": "https://example.com",
            "result_ids": ["a", "b"],
            "source_values": ["Example.com", "https://www.example.com/"],
        }
    ]
    assert report["duplicate_result_ids"] == ["b"]
    assert report["unique_source_count"] == 1
