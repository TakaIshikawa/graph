from __future__ import annotations

from graph.rag.answer_prescriptive_claims import audit_answer_prescriptive_claims


def test_supported_recommendation_has_nearby_citation_and_explicit_evidence():
    answer = "For production login, you should use OAuth [1]."
    report = audit_answer_prescriptive_claims(
        answer,
        evidence_spans=[
            {
                "id": "ev1",
                "citation_id": "[1]",
                "text": "For production login, use OAuth because it supports delegated authorization.",
            }
        ],
    )

    assert report["claim_count"] == 1
    assert report["supported_claim_count"] == 1
    assert report["flagged_claims"] == []
    assert report["claims"][0]["evidence_ids"] == ["ev1"]


def test_unsupported_directive_reports_offsets_and_reason():
    answer = "Summarize the notes. Avoid password reuse."
    report = audit_answer_prescriptive_claims(answer)

    assert report["unsupported_claim_count"] == 1
    claim = report["flagged_claims"][0]
    assert claim["support_status"] == "unsupported"
    assert claim["prescriptive_cues"] == ["avoid"]
    assert claim["start"] == answer.index("Avoid")
    assert answer[claim["start"] : claim["end"]] == "Avoid password reuse."
    assert claim["reasons"] == ["no_nearby_citation_or_evidence"]


def test_nearby_citation_without_matching_evidence_is_weak():
    answer = "You must switch to passkeys [policy]."
    report = audit_answer_prescriptive_claims(
        answer,
        evidence_spans=[{"id": "policy", "text": "The policy describes account recovery timelines."}],
        citation_spans=[{"id": "policy", "start": answer.index("[policy]"), "end": answer.index("[policy]") + len("[policy]")}],
    )

    assert report["weak_claim_count"] == 1
    assert report["flagged_claims"][0]["support_status"] == "weak"
    assert set(report["flagged_claims"][0]["reasons"]) == {"evidence_not_prescriptive", "low_evidence_overlap"}


def test_descriptive_only_answers_have_no_prescriptive_claims():
    report = audit_answer_prescriptive_claims(
        "The migration finished on Tuesday. The old service remains online.",
        evidence_spans=[{"text": "The old service remains online."}],
    )

    assert report["claim_count"] == 0
    assert report["flagged_claims"] == []
    assert report["warnings"] == []
