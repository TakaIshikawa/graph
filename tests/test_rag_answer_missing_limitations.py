from __future__ import annotations

from graph.rag.answer_missing_limitations import audit_answer_missing_limitations


def test_answer_missing_limitations_detects_sparse_evidence():
    audit = audit_answer_missing_limitations("Confident answer.", [{"source_id": "a", "date": "2025-01-01"}], now="2025-06-01")

    assert "sparse evidence" in audit["evidence_risks"]
    assert "single-source evidence" in audit["evidence_risks"]
    assert audit["missing_limitations"] == audit["evidence_risks"]


def test_answer_missing_limitations_detects_old_evidence_dates():
    audit = audit_answer_missing_limitations(
        "Answer.",
        [{"source_id": "a", "date": "2020-01-01"}, {"source_id": "b", "date": "2020-02-01"}],
        now="2025-06-01",
    )

    assert audit["evidence_risks"] == ["old evidence dates"]
    assert audit["recommended_caveats"] == ["Mention that fresher evidence may change the conclusion."]


def test_answer_missing_limitations_detects_low_confidence_metadata():
    audit = audit_answer_missing_limitations(
        "Answer.",
        [{"source_id": "a", "date": "2025-01-01", "confidence": 0.4}, {"source_id": "b", "date": "2025-01-02"}],
        now="2025-06-01",
    )

    assert audit["evidence_risks"] == ["low confidence metadata"]


def test_answer_missing_limitations_scores_well_supported_answer():
    audit = audit_answer_missing_limitations(
        "Answer.",
        [{"source_id": "a", "date": "2025-01-01"}, {"source_id": "b", "date": "2025-01-02"}],
        now="2025-06-01",
    )

    assert audit == {
        "missing_limitations": [],
        "evidence_risks": [],
        "recommended_caveats": [],
        "limitation_score": 1.0,
    }
