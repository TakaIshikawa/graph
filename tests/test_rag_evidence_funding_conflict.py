from __future__ import annotations

from graph.rag.evidence_funding_conflict import detect_evidence_funding_conflicts


def test_evidence_funding_conflict_detects_conflicts_and_funders():
    result = detect_evidence_funding_conflicts(
        [
            {"id": "e1", "text": "The trial was funded by Acme Pharma and the author is a consultant for Acme."},
            {"id": "e2", "metadata": {"disclosure": "No conflicts declared."}},
        ]
    )

    assert result["conflict_evidence"] == [{"evidence_id": "e1", "conflict_cues": ["funded_by", "consulting_conflict"]}]
    assert result["funder_mentions"] == [{"evidence_id": "e1", "mention": "funded by Acme Pharma and the author is a consultant for Acme"}]
    assert result["disclosure_evidence"] == [{"evidence_id": "e2", "disclosure_type": "no_conflicts_declared"}]
    assert result["warnings"] == ["funding_or_conflict_disclosure_present"]
    assert result["confidence"] == 0.85


def test_evidence_funding_conflict_returns_empty_for_no_cues():
    assert detect_evidence_funding_conflicts([{"id": "e1", "text": "Independent methods report."}]) == {
        "conflict_evidence": [],
        "disclosure_evidence": [],
        "funder_mentions": [],
        "warnings": [],
        "confidence": 0.0,
    }
