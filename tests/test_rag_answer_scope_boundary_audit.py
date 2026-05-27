from __future__ import annotations

from graph.rag.answer_scope_boundary_audit import audit_answer_scope_boundaries


def test_answer_scope_boundary_allows_query_constrained_answers():
    audit = audit_answer_scope_boundaries("Acme should focus on 2024 retention in the US.", "Acme 2024 retention in the US")

    assert audit["issue_count"] == 0
    assert audit["severity"] == "none"


def test_answer_scope_boundary_flags_entity_drift():
    audit = audit_answer_scope_boundaries("Acme is stable. Globex changes the risk profile.", "Acme risk profile")

    assert audit["summary"]["entity_drift"] == 1
    assert audit["issues"][0]["reason"] == "entity_drift"
    assert audit["issues"][0]["sentence"] == "Globex changes the risk profile."


def test_answer_scope_boundary_flags_temporal_drift():
    audit = audit_answer_scope_boundaries("The 2023 pattern is not enough; 2025 data changes it.", "2023 pattern")

    assert audit["summary"]["temporal_drift"] == 1
    assert audit["issues"][0]["reason"] == "temporal_drift"


def test_answer_scope_boundary_flags_jurisdiction_drift():
    audit = audit_answer_scope_boundaries("The same rule also applies in the EU.", "US rule")

    assert audit["summary"]["jurisdiction_drift"] == 1
    assert audit["severity"] == "high"
