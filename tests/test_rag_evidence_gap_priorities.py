from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph.rag.evidence_gap_priorities import prioritize_evidence_gaps


def test_evidence_gap_priorities_normalizes_and_sorts_records():
    report = prioritize_evidence_gaps(
        [
            {"id": "b", "severity": "low", "missing_field": "citation", "claim": "Claim B"},
            SimpleNamespace(id="a", level="critical", field="date", source="docs"),
            {"id": "c", "metadata": {"severity": "high", "gap_type": "source", "statement": "Claim C"}},
        ],
        max_items=2,
    )

    assert report["total_gaps"] == 3
    assert [item["gap_id"] for item in report["priorities"]] == ["a", "c"]
    assert report["priorities"][0]["recommended_action"] == "find_date_support"
    assert report["priorities"][1]["recommended_action"] == "retrieve_primary_source"


def test_evidence_gap_priorities_validates_max_items():
    with pytest.raises(ValueError, match="max_items"):
        prioritize_evidence_gaps([], max_items=0)
