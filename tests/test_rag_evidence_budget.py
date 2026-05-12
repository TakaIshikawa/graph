from __future__ import annotations

import pytest

from graph.rag import allocate_evidence_budget


def test_evidence_budget_prefers_coverage_and_source_diversity():
    payload = allocate_evidence_budget(
        [
            {
                "id": "same-source-high",
                "title": "Vector search",
                "content": "Vector search search",
                "source_project": "notes",
                "score": 0.96,
            },
            {
                "id": "diverse-coverage",
                "title": "Graph citations",
                "content": "citation graph evidence",
                "source_project": "papers",
                "score": 0.82,
                "url": "https://example.test/paper",
                "published_at": "2026-04-01",
            },
            {
                "id": "same-source-low",
                "title": "Vector notes",
                "content": "search notes",
                "source_project": "notes",
                "score": 0.75,
            },
        ],
        "vector search citation graph",
        max_results=2,
        min_sources=2,
    )

    assert {item["result_id"] for item in payload["selected_results"]} == {
        "same-source-high",
        "diverse-coverage",
    }
    assert payload["covered_terms"] == ["citation", "graph", "search", "vector"]
    assert payload["source_counts"] == {"notes": 1, "papers": 1}
    assert payload["omitted_result_ids"] == ["same-source-low"]


def test_evidence_budget_never_exceeds_max_results_and_reports_stats():
    payload = allocate_evidence_budget(
        [
            {"id": "a", "content": "alpha", "score": 0.1},
            {"id": "b", "content": "beta", "score": 0.2},
            {"id": "c", "content": "gamma", "score": 0.3},
        ],
        "alpha beta gamma",
        max_results=2,
    )

    assert len(payload["selected_results"]) == 2
    assert payload["stats"] == {
        "total_results": 3,
        "selected_count": 2,
        "omitted_count": 1,
        "max_results": 2,
        "min_sources": 2,
        "query_term_count": 3,
    }


def test_evidence_budget_supports_nested_units_and_tuple_scores():
    payload = allocate_evidence_budget(
        [
            (
                {
                    "unit": {
                        "id": "nested",
                        "title": "Hybrid retrieval",
                        "metadata": {
                            "source_project": "archive",
                            "keywords": ["semantic ranking"],
                            "source_url": "https://archive.example/item",
                            "updated_at": "2026-05-01",
                        },
                    }
                },
                0.7,
            )
        ],
        "semantic ranking retrieval",
    )

    assert payload["selected_results"] == [
        {
            "rank": 1,
            "result_id": "nested",
            "title": "Hybrid retrieval",
            "source_project": "archive",
            "score": 0.7,
            "covered_terms": ["ranking", "retrieval", "semantic"],
            "has_citation": True,
            "has_date": True,
        }
    ]


@pytest.mark.parametrize("kwargs", [{"max_results": 0}, {"min_sources": 0}, {"max_results": True}])
def test_evidence_budget_validates_limits(kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        allocate_evidence_budget([], "query", **kwargs)
