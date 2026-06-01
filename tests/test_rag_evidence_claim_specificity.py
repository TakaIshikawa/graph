from graph.rag.evidence_claim_specificity import score_evidence_claim_specificity


def test_high_specificity_evidence_scores_multiple_features():
    summary = score_evidence_claim_specificity(
        [{"id": "a", "text": 'Acme Cloud reported "latency fell" by 24% in 2025 compared with 2024.'}]
    )

    assert summary["record_count"] == 1
    assert summary["high_specificity_count"] == 1
    assert summary["samples"][0]["features"] == ["comparison", "date", "named_entity", "numeric_value", "quoted_span"]


def test_vague_evidence_is_low_specificity():
    summary = score_evidence_claim_specificity([{"id": "v", "text": "It is generally useful."}])

    assert summary["low_specificity_count"] == 1
    assert summary["average_specificity_score"] == 0.0


def test_sample_limiting_is_deterministic():
    summary = score_evidence_claim_specificity([{"id": "a", "text": "2025"}, {"id": "b", "text": "2026"}], sample_limit=1)

    assert summary["samples"] == [{"result_id": "a", "score": 2, "features": ["date", "numeric_value"]}]
