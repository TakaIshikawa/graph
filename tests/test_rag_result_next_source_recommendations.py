from __future__ import annotations

from graph.rag.result_next_source_recommendations import recommend_result_next_sources


def test_result_next_source_recommendations_empty_results_rank_missing_categories():
    result = recommend_result_next_sources([], ["local"])

    assert result["recommendations"][0]["source_category"] == "local_source"
    assert result["recommendations"][0]["reason_codes"] == ["MISSING_LOCAL"]
    assert len(result["recommendations"]) == 5


def test_result_next_source_recommendations_detects_existing_coverage():
    result = recommend_result_next_sources(
        [
            {"id": "a", "source_type": "official primary dataset", "date": "2026-01-01"},
            {"id": "b", "text": "However, a local methodology critique disputed the sample."},
        ]
    )

    assert result["coverage"] == {
        "primary": True,
        "recent": True,
        "dissenting": True,
        "local": True,
        "methodology": True,
    }
    assert result["recommendations"] == []
