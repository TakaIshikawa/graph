from graph.rag.result_source_type_balance import analyze_result_source_type_balance


def test_uses_metadata_before_url_heuristics_and_computes_balance():
    summary = analyze_result_source_type_balance(
        [
            {"metadata": {"source_type": "paper"}, "url": "https://github.com/org/repo"},
            {"url": "https://docs.example.com/api"},
            {"url": "https://www.kaggle.com/datasets/x/y"},
            {"url": "https://stackoverflow.com/questions/1"},
        ]
    )

    assert summary == {
        "total_results": 4,
        "source_type_counts": {"dataset": 1, "documentation": 1, "forum": 1, "paper": 1},
        "dominant_source_type": "documentation",
        "diversity_score": 0.375,
    }
