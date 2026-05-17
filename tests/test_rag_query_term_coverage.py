from __future__ import annotations

from dataclasses import dataclass

from graph.rag import score_query_term_coverage


@dataclass
class Result:
    id: str
    title: str
    content: str
    tags: list[str]


def test_query_term_coverage_reports_full_coverage_case_insensitively():
    rows = score_query_term_coverage(
        "Battery battery Storage Policy",
        [
            Result(
                id="full",
                title="Battery storage policy",
                content="Operational notes",
                tags=["Energy"],
            )
        ],
    )

    assert rows == [
        {
            "result_id": "full",
            "matched_terms": ["battery", "storage", "policy"],
            "missing_terms": [],
            "coverage_score": 1.0,
        }
    ]


def test_query_term_coverage_reports_partial_coverage_from_text_and_tags():
    rows = score_query_term_coverage(
        "battery retention policy",
        [
            {
                "id": "partial",
                "title": "Battery operations",
                "metadata": {"tags": [{"tag": "Retention"}]},
            }
        ],
    )

    assert rows == [
        {
            "result_id": "partial",
            "matched_terms": ["battery", "retention"],
            "missing_terms": ["policy"],
            "coverage_score": 0.667,
        }
    ]


def test_query_term_coverage_reports_zero_coverage_and_index_fallback():
    rows = score_query_term_coverage(
        "battery storage",
        [{"title": "Unrelated climate note", "content": "Weather observations"}],
    )

    assert rows == [
        {
            "result_id": "result-1",
            "matched_terms": [],
            "missing_terms": ["battery", "storage"],
            "coverage_score": 0.0,
        }
    ]
