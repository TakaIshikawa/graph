from __future__ import annotations

from graph.rag.evidence_saturation import analyze_evidence_saturation


def test_evidence_saturation_detects_plateau_after_repeated_context():
    report = analyze_evidence_saturation(
        [
            {"id": "a", "domain": "one.test", "content": "hybrid search ranking", "tags": ["bm25"]},
            {"id": "b", "domain": "one.test", "content": "hybrid search ranking", "tags": ["bm25"]},
            {"id": "c", "domain": "two.test", "content": "vector reranking latency", "tags": ["latency"]},
        ],
        query="hybrid search ranking latency",
    )

    assert report["result_count"] == 3
    assert report["first_plateau_index"] == 1
    assert "evidence_plateau" in report["warnings"]
    assert report["marginal_gains"][0]["marginal_gain"] > report["marginal_gains"][1]["marginal_gain"]


def test_evidence_saturation_empty_results():
    assert analyze_evidence_saturation([])["warnings"] == ["no_results"]
