from __future__ import annotations

from graph.rag.evidence_quoted_claim_alignment import audit_evidence_quoted_claim_alignment


def test_quoted_claim_alignment_empty_claims():
    assert audit_evidence_quoted_claim_alignment([], []) == {"alignments": []}


def test_quoted_claim_alignment_scores_aligned_claim():
    result = audit_evidence_quoted_claim_alignment(
        [{"id": "c1", "claim_text": "Revenue rose 12% in 2024.", "evidence_id": "e1"}],
        [{"id": "e1", "quote": "The report says revenue rose 12% in 2024."}],
    )

    row = result["alignments"][0]
    assert row["alignment_status"] == "aligned"
    assert row["numeric_agreement"] == 1.0
    assert row["polarity_mismatch"] is False


def test_quoted_claim_alignment_flags_numeric_and_polarity_conflict():
    result = audit_evidence_quoted_claim_alignment(
        [{"id": "c1", "claim_text": "The intervention improved outcomes by 30%.", "evidence_id": "e1"}],
        [{"id": "e1", "quote": "The intervention did not improve outcomes by 10%."}],
    )

    row = result["alignments"][0]
    assert row["alignment_status"] == "conflicting"
    assert row["numeric_agreement"] == 0.0
    assert row["polarity_mismatch"] is True
