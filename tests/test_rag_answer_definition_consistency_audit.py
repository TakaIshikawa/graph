from __future__ import annotations

from graph.rag.answer_definition_consistency_audit import audit_answer_definition_consistency


def test_answer_definition_consistency_marks_supported_definition():
    rows = audit_answer_definition_consistency("Latency is defined as response delay.", [{"text": "Latency is defined as response delay."}])

    assert rows == [{"term": "latency", "definition_sentence": "Latency is defined as response delay.", "evidence_match_count": 1, "conflicting_evidence_count": 0, "severity": None}]


def test_answer_definition_consistency_marks_unsupported_and_deduplicates():
    rows = audit_answer_definition_consistency("SLO means service target. SLO means service target.", [])

    assert rows == [{"term": "slo", "definition_sentence": "SLO means service target.", "evidence_match_count": 0, "conflicting_evidence_count": 0, "severity": "unsupported"}]


def test_answer_definition_consistency_marks_conflicting_definition():
    rows = audit_answer_definition_consistency("Latency refers to response delay.", ["Latency is not response delay; it differs by context."])

    assert rows[0]["conflicting_evidence_count"] == 1
    assert rows[0]["severity"] == "conflict"
