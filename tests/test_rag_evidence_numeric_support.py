from __future__ import annotations

from graph.rag.evidence_numeric_support import audit_evidence_numeric_support


def test_finds_supported_and_unsupported_numeric_claims():
    report = audit_evidence_numeric_support("Revenue rose 10% to $2,500 in 2025.", [{"id": "r1", "text": "Revenue rose 10 percent to $2500."}])

    assert [row["matched_text"] for row in report["supported_numbers"]] == ["10%", "$2,500"]
    assert [row["matched_text"] for row in report["unsupported_numbers"]] == ["2025"]
    assert report["reason_counts"] == {"missing_numeric_evidence": 1}


def test_currency_and_ranges_are_structured():
    report = audit_evidence_numeric_support("Costs ranged from $10-20 and margin was 2.5.", [{"id": "r1", "text": "Costs ranged from $10 to $20."}])

    assert report["answer_numbers"][0]["number_type"] == "range"
    assert report["answer_numbers"][0]["values"] == ["$10", "20"]
    assert any(row["number_type"] == "decimal" for row in report["answer_numbers"])


def test_empty_inputs_return_deterministic_warnings():
    report = audit_evidence_numeric_support("", [])

    assert report["warnings"] == ["no_answer", "no_results"]
    assert report["answer_numbers"] == []
