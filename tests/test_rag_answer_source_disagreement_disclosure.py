from __future__ import annotations

from graph.rag import audit_answer_source_disagreement_disclosure


def test_answer_source_disagreement_disclosure_flags_missing_acknowledgement():
    report = audit_answer_source_disagreement_disclosure(
        "The answer is settled.",
        [{"id": "r1", "content": "Sources differ and the evidence is disputed."}],
    )

    assert report["has_evidence_disagreement"] is True
    assert report["answer_discloses_disagreement"] is False
    assert report["missing_disclosure"] is True
    assert report["cue_counts"]["disputed"] == 1
    assert report["samples"][0] == {"result_id": "r1", "cue": "disputed"}


def test_answer_source_disagreement_disclosure_accepts_answer_acknowledgement():
    report = audit_answer_source_disagreement_disclosure(
        "The evidence is mixed.",
        [{"id": "r1", "snippet": "Conflicting estimates remain."}],
    )

    assert report["missing_disclosure"] is False
