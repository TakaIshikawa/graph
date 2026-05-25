from __future__ import annotations

from graph.rag.result_retrieval_overlap import analyze_retrieval_overlap


def test_retrieval_overlap_accepts_ids_and_result_dicts():
    result = analyze_retrieval_overlap(
        {
            "bm25": ["a", "b", "c"],
            "vector": [{"id": "b"}, {"id": "c"}, {"id": "d"}],
            "hybrid": ["c", "e"],
        }
    )

    assert result["overlapping_ids"] == ["b", "c"]
    assert result["consensus_ids"] == ["c"]
    assert result["unique_ids_by_strategy"]["vector"] == ["d"]
    assert result["pairwise_overlap"][0]["overlap_ratio"] == 0.5


def test_retrieval_overlap_handles_empty_strategies():
    result = analyze_retrieval_overlap({"empty": [], "full": ["a"]})

    assert result["pairwise_overlap"][0]["overlap_ratio"] == 0.0
    assert result["unique_ids_by_strategy"]["full"] == ["a"]
