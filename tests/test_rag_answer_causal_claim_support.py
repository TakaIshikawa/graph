from __future__ import annotations

from graph.rag.answer_causal_claim_support import audit_answer_causal_claim_support


def test_extracts_causal_sentences_and_cues():
    result = audit_answer_causal_claim_support("The rollout caused churn. It was popular.", [])

    assert result["causal_claims"][0]["sentence"] == "The rollout caused churn."
    assert result["causal_claims"][0]["cue_words"] == ["caused"]
    assert result["unsupported_causal_claims"]


def test_strong_causal_evidence_clears_unsupported_claims():
    result = audit_answer_causal_claim_support(
        "Training leads to faster resolution.",
        [{"text": "A randomized controlled longitudinal study tested the mechanism."}],
    )

    assert result["unsupported_causal_claims"] == []
    assert result["support_summary"]["strong_support_count"] == 1


def test_correlational_only_evidence_keeps_claim_flagged():
    result = audit_answer_causal_claim_support("Usage drives retention.", [{"text": "Observational correlation was reported."}])

    assert result["unsupported_causal_claims"]
    assert result["support_summary"]["correlational_only_count"] == 1
