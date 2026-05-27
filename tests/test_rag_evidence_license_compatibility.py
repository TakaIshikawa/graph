from __future__ import annotations

from graph.rag.evidence_license_compatibility import audit_evidence_license_compatibility


def test_audit_evidence_license_compatibility_uses_aliases_and_defaults():
    summary = audit_evidence_license_compatibility(
        [
            {"id": "a", "license": "CC-BY"},
            {"id": "b", "rights": "all rights reserved"},
            {"id": "c", "metadata": {"license": "unknown"}},
            {"id": "d"},
        ]
    )

    assert summary["compatible_count"] == 1
    assert summary["incompatible_count"] == 1
    assert summary["unknown_count"] == 1
    assert summary["missing_count"] == 1
    assert summary["compatible_evidence_ids"] == ["a"]


def test_audit_evidence_license_compatibility_allows_override():
    assert audit_evidence_license_compatibility([{"id": "x", "license": "proprietary"}], ["proprietary"])["compatible_count"] == 1
