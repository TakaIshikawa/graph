from __future__ import annotations

from graph.rag.evidence_accessibility_summary import summarize_evidence_accessibility


def test_evidence_accessibility_summary_buckets_access_fields():
    summary = summarize_evidence_accessibility(
        [
            {"id": "open", "access": "open"},
            {"id": "pay", "metadata": {"paywalled": "yes"}},
            {"id": "login", "login_required": True},
            {"id": "priv", "access_status": "restricted"},
            {"id": "miss"},
        ]
    )

    assert summary["bucket_counts"] == {"open": 1, "restricted": 1, "paywalled": 1, "login_required": 1, "missing_access": 1, "unknown": 0}
    assert summary["restricted_ids"] == ["login", "pay", "priv"]
    assert summary["missing_access_count"] == 1
    assert summary["open_ratio"] == 0.2


def test_evidence_accessibility_summary_handles_empty_input():
    assert summarize_evidence_accessibility([])["open_ratio"] == 0.0
