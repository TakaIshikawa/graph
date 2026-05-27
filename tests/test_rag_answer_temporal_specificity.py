from graph.rag.answer_temporal_specificity import audit_answer_temporal_specificity


def test_temporal_specificity_counts_dated_and_vague_claims():
    report = audit_answer_temporal_specificity(
        "Currently adoption is rising. Revenue increased in 2025. The latest release lands soon."
    )

    assert report == {
        "temporal_claim_count": 3,
        "dated_claim_count": 1,
        "vague_temporal_claim_count": 2,
        "specificity_ratio": 0.3333,
        "findings": [
            {"type": "vague_temporal_claim", "snippet": "Currently adoption is rising."},
            {"type": "vague_temporal_claim", "snippet": "The latest release lands soon."},
        ],
    }


def test_temporal_specificity_treats_iso_dates_as_specific():
    report = audit_answer_temporal_specificity("The policy changed on 2025-04-30. The method is stable.")

    assert report["dated_claim_count"] == 1
    assert report["vague_temporal_claim_count"] == 0
