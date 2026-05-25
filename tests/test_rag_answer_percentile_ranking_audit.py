from __future__ import annotations

from graph.rag.answer_percentile_ranking_audit import audit_answer_percentile_rankings


def test_percentile_ranking_flags_unsupported_claims():
    result = audit_answer_percentile_rankings(
        "The model is in the top 10% and ranked #1 for accuracy.",
        [{"id": "e1", "text": "The model was evaluated on accuracy."}],
    )

    finding = result["findings"][0]
    assert finding["claim_text"] == "The model is in the top 10% and ranked #1 for accuracy."
    assert finding["ranking_terms"] == ["top 10%", "ranked #1"]
    assert finding["supporting_evidence_ids"] == []
    assert finding["reason_codes"] == ["ranking_claim_without_ranking_evidence"]


def test_percentile_ranking_treats_rank_evidence_as_support():
    result = audit_answer_percentile_rankings(
        "The model is in the highest quartile.",
        [{"id": "e1", "text": "Benchmark table: highest quartile for accuracy."}],
    )

    assert result["findings"] == []
    assert result["ranking_evidence_count"] == 1
