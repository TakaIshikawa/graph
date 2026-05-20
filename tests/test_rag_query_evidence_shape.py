from __future__ import annotations

import pytest

from graph.rag.query_evidence_shape import classify_query_evidence_shape


@pytest.mark.parametrize(
    ("query", "shape"),
    [
        ("Compare SQLite vs Postgres for local search", "comparison"),
        ("Timeline of model releases after 2024", "timeline"),
        ("Why does retrieval fail with this error?", "diagnostic"),
        ("How to configure hybrid search", "how_to"),
        ("What is the best vector database?", "opinion_or_preference"),
        ("What is BM25?", "fact_lookup"),
    ],
)
def test_query_evidence_shape_recognizes_cues(query, shape):
    result = classify_query_evidence_shape(query, result_count=0)

    assert result["shape"] == shape
    assert result["recommended_min_results"] >= 1
    assert result["warnings"] == ["insufficient_result_count"]


def test_query_evidence_shape_rejects_blank_queries():
    with pytest.raises(ValueError, match="blank"):
        classify_query_evidence_shape("   ")
