from __future__ import annotations

from graph.rag.answer_claim_sequence import audit_answer_claim_sequence


def test_answer_claim_sequence_scores_chronological_answers():
    audit = audit_answer_claim_sequence("In 2020 it launched. In 2021 it scaled. In 2023 it matured.")

    assert [row["year"] for row in audit["ordered_claims"]] == [2020, 2021, 2023]
    assert audit["out_of_order_claims"] == []
    assert audit["sequence_score"] == 1.0


def test_answer_claim_sequence_flags_reverse_chronological_answers():
    audit = audit_answer_claim_sequence("In 2023 it matured. In 2021 it scaled. In 2020 it launched.")

    assert [row["year"] for row in audit["out_of_order_claims"]] == [2021, 2020]
    assert audit["chronology_hint"] == "reorder dated claims chronologically"


def test_answer_claim_sequence_handles_undated_answers():
    audit = audit_answer_claim_sequence("The program launched and later expanded.")

    assert audit["ordered_claims"] == []
    assert audit["out_of_order_claims"] == []
    assert audit["sequence_score"] == 1.0
    assert audit["chronology_hint"] == "no dated claims detected"
