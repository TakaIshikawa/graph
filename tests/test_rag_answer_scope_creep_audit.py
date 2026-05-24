from __future__ import annotations

from graph.rag.answer_scope_creep_audit import audit_answer_scope_creep


def test_answer_scope_creep_empty_answer():
    assert audit_answer_scope_creep("Compare Japan solar policy in 2024", "")["findings"] == []


def test_answer_scope_creep_flags_unrelated_entities_dates_and_geography():
    result = audit_answer_scope_creep(
        "Compare Japan solar policy in 2024.",
        "Japan expanded solar incentives in 2024. Brazil wind policy changed in 2021.",
    )

    finding = result["findings"][0]
    assert finding["span_text"] == "Brazil wind policy changed in 2021."
    assert "new_entity" in finding["reason_codes"]
    assert "new_date" in finding["reason_codes"]
    assert "new_geography" in finding["reason_codes"]


def test_answer_scope_creep_flags_new_task_intent():
    result = audit_answer_scope_creep(
        "Summarize Japan solar policy.",
        "Calculate Japan solar policy costs.",
    )

    assert result["findings"][0]["reason_codes"] == ["new_task_intent"]
