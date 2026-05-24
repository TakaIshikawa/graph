from __future__ import annotations

from graph.rag.answer_missing_counterexample_audit import audit_answer_missing_counterexamples


def test_answer_missing_counterexamples_empty_inputs():
    assert audit_answer_missing_counterexamples("", []) == {
        "findings": [],
        "counterexample_evidence_count": 0,
    }


def test_answer_missing_counterexamples_flags_broad_unacknowledged_claim():
    result = audit_answer_missing_counterexamples(
        "The therapy always improves sleep quality.",
        [
            {"id": "e1", "text": "However, the therapy did not improve sleep for older participants."},
            {"id": "e2", "text": "The study measured sleep quality."},
        ],
    )

    finding = result["findings"][0]
    assert finding["claim_text"] == "The therapy always improves sleep quality."
    assert finding["counterexample_evidence_ids"] == ["e1"]
    assert finding["severity"] == "medium"


def test_answer_missing_counterexamples_respects_answer_acknowledgement():
    result = audit_answer_missing_counterexamples(
        "The therapy improves sleep quality, although some groups did not benefit.",
        [{"id": "e1", "text": "But older participants showed no evidence of improvement."}],
    )

    assert result["findings"] == []
