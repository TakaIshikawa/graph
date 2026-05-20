from __future__ import annotations

from graph.rag.context_claim_coverage import analyze_context_claim_coverage


def test_context_claim_coverage_reports_full_partial_and_missing_support():
    report = analyze_context_claim_coverage(
        [
            {"id": "full", "text": "hybrid search improves ranking"},
            {"id": "partial", "text": "latency depends on reranking cache"},
            "unsupported privacy guarantee",
        ],
        [
            {"id": "r1", "content": "hybrid search improves ranking with bm25"},
            {"id": "r2", "content": "reranking adds latency"},
        ],
    )

    rows = report["claims"]
    assert rows[0]["coverage_ratio"] == 1.0
    assert rows[0]["supporting_result_ids"] == ["r1"]
    assert rows[1]["warnings"] == ["weak_support"]
    assert rows[2]["warnings"] == ["missing_support"]
    assert "missing_support" in report["warnings"]
