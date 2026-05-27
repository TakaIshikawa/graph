from __future__ import annotations

from graph.rag.context_evidence_role_balance import analyze_context_evidence_role_balance


def test_context_evidence_role_balance_counts_balanced_contexts():
    summary = analyze_context_evidence_role_balance([{"role": "primary"}, {"metadata": {"evidence_role": "secondary"}}, {"role": "background"}])

    assert summary["role_counts"] == {"background": 1, "primary": 1, "secondary": 1}
    assert summary["imbalance_flags"] == []


def test_context_evidence_role_balance_flags_missing_primary_and_background_heavy():
    summary = analyze_context_evidence_role_balance([{"role": "background"}, {"role": "background"}, {}])

    assert summary["missing_role_count"] == 1
    assert summary["dominant_role"] == "background"
    assert summary["imbalance_flags"] == ["absent_primary_evidence", "background_heavy_context"]
