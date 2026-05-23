from __future__ import annotations

from graph.rag.evidence_unit_consistency import check_evidence_unit_consistency


def test_evidence_unit_consistency_scores_consistent_units():
    payload = check_evidence_unit_consistency([{"id": "a", "text": "$10"}, {"id": "b", "text": "USD 12"}])

    assert payload["inconsistent_groups"] == []
    assert payload["consistency_score"] == 1.0


def test_evidence_unit_consistency_flags_mixed_units():
    payload = check_evidence_unit_consistency([{"id": "a", "text": "$10 and 5 km"}, {"id": "b", "text": "€12 and 3 miles"}])

    assert payload["inconsistent_groups"] == [
        {"group": "currency", "units": ["EUR", "USD"]},
        {"group": "distance", "units": ["km", "mi"]},
    ]
    assert len(payload["normalization_hints"]) == 2


def test_evidence_unit_consistency_handles_no_units():
    payload = check_evidence_unit_consistency([{"id": "a", "text": "No measurements here."}])

    assert payload == {
        "detected_units": [],
        "inconsistent_groups": [],
        "normalization_hints": [],
        "consistency_score": 1.0,
    }
