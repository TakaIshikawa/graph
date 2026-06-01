from graph.rag.evidence_benchmark_signal import analyze_evidence_benchmark_signals


def test_detects_benchmark_metrics_and_dataset_terms():
    summary = analyze_evidence_benchmark_signals(
        [
            {"id": "a", "text": "Benchmark baseline reports accuracy and F1 on the validation set."},
            {"id": "b", "metadata": {"note": "Leaderboard score on dataset v2"}},
        ]
    )

    assert summary["benchmark_evidence_count"] == 2
    assert summary["metric_terms"] == ["accuracy", "f1", "score"]
    assert summary["dataset_terms"] == ["dataset", "validation_set"]
    assert summary["benchmark_samples"][0]["source_id"] == "a"


def test_non_benchmark_evidence_returns_empty_terms():
    assert analyze_evidence_benchmark_signals([{"text": "General overview."}]) == {
        "benchmark_evidence_count": 0,
        "metric_terms": [],
        "dataset_terms": [],
        "benchmark_samples": [],
    }
