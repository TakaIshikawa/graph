from __future__ import annotations

import pytest

from graph.rag import explain_rag_results


def test_result_explanations_assign_match_and_metadata_labels():
    rows = explain_rag_results(
        [
            {
                "id": "r1",
                "title": "Vector search notes",
                "content": "Embeddings improve recall.",
                "tags": ["semantic-ranking"],
                "metadata": {
                    "keywords": ["citation graph"],
                    "url": "https://example.test/r1",
                    "published_at": "2026-04-01",
                },
                "score": 0.91,
            }
        ],
        "vector recall semantic citation",
    )

    assert rows == [
        {
            "result_id": "r1",
            "labels": [
                "title_match",
                "content_match",
                "tag_match",
                "metadata_match",
                "cited",
                "dated",
                "high_score",
            ],
            "matched_terms": ["citation", "recall", "semantic", "vector"],
            "evidence_summary": "matched citation, recall, semantic, vector; title match; content match; tag match",
        }
    ]


def test_result_explanations_marks_weak_match_but_preserves_order():
    rows = explain_rag_results(
        [
            {"id": "first", "title": "Unrelated", "score": 0.9},
            {"id": "second", "title": "Graph evidence"},
        ],
        "graph",
    )

    assert [row["result_id"] for row in rows] == ["first", "second"]
    assert rows[0]["labels"] == ["high_score", "weak_match"]
    assert rows[0]["matched_terms"] == []
    assert rows[1]["labels"] == ["title_match"]


def test_result_explanations_supports_nested_units_and_tuple_scores():
    rows = explain_rag_results(
        [
            (
                {
                    "unit": {
                        "id": "nested",
                        "title": "Hybrid retrieval",
                        "metadata": {
                            "keywords": [{"term": "semantic expansion"}],
                            "source_url": "https://archive.example/item",
                            "updated_at": "2026-05-01",
                        },
                    }
                },
                0.8,
            )
        ],
        "semantic retrieval",
    )

    assert rows[0]["result_id"] == "nested"
    assert rows[0]["labels"] == ["title_match", "metadata_match", "cited", "dated", "high_score"]
    assert rows[0]["matched_terms"] == ["retrieval", "semantic"]


def test_result_explanations_limits_summary_reasons():
    rows = explain_rag_results(
        [{"id": "r1", "title": "alpha", "content": "beta", "tags": ["gamma"]}],
        "alpha beta gamma",
        max_reasons=2,
    )

    assert rows[0]["evidence_summary"] == "matched alpha, beta, gamma; title match"


@pytest.mark.parametrize("max_reasons", [0, -1, True, "2"])
def test_result_explanations_validates_max_reasons(max_reasons):
    with pytest.raises(ValueError, match="max_reasons must be a positive integer"):
        explain_rag_results([], "query", max_reasons=max_reasons)
