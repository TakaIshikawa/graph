from __future__ import annotations

from graph.rag.answer_policy_exception_audit import audit_answer_policy_exceptions


def test_policy_exception_flags_rule_claim_that_omits_exception_evidence():
    result = audit_answer_policy_exceptions(
        "Applicants must submit the form and are required to pay the fee.",
        [{"id": "policy", "text": "A waiver is available case-by-case for financial hardship."}],
    )

    assert result["exception_evidence_count"] == 1
    assert result["findings"] == [
        {
            "claim_text": "Applicants must submit the form and are required to pay the fee.",
            "exception_evidence_ids": ["policy"],
            "severity": "medium",
            "reason_codes": ["rule_claim_omits_available_exception"],
        }
    ]


def test_policy_exception_does_not_flag_when_answer_acknowledges_exceptions():
    result = audit_answer_policy_exceptions(
        "Applicants must submit the form unless a waiver applies.",
        [{"id": "policy", "text": "A waiver is available for hardship."}],
    )

    assert result["findings"] == []
