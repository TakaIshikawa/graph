from __future__ import annotations

from graph.rag.answer_conditional_claims import audit_answer_conditional_claims


def test_answer_conditional_claim_audit_marks_supported_conditionals():
    rows = audit_answer_conditional_claims(
        "If latency exceeds 200ms, you should reduce batch size.",
        [{"content": "if latency exceeds 200ms you should reduce batch size"}],
    )

    assert rows[0]["supported"] is True
    assert rows[0]["severity"] == "none"


def test_answer_conditional_claim_audit_marks_unsupported_conditionals():
    rows = audit_answer_conditional_claims("When storage is full, choose archive mode.", [{"content": "No recommendation."}])

    assert rows[0]["condition_phrase"] == "when storage is full"
    assert rows[0]["supported"] is False
    assert rows[0]["severity"] == "medium"


def test_answer_conditional_claim_audit_ignores_non_conditional_prose():
    assert audit_answer_conditional_claims("Archive mode reduces storage pressure.") == []


def test_answer_conditional_claim_audit_returns_multiple_conditions():
    rows = audit_answer_conditional_claims("If demand rises and when queues grow, consider scaling workers.")

    assert [row["condition_phrase"] for row in rows] == ["if demand rises and when queues grow", "when queues grow"]
