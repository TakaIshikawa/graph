from __future__ import annotations

from graph.rag.evidence_claim_recency_mismatch import analyze_evidence_claim_recency_mismatch


def test_flags_current_claim_supported_only_by_stale_evidence():
    report = analyze_evidence_claim_recency_mismatch(
        "The vendor is currently the market leader.",
        [{"id": "old", "content": "Market share report", "metadata": {"published_at": "2024-01-01"}}],
        now="2026-05-25",
    )

    assert report["has_current_claim"] is True
    assert report["evidence"][0]["age_days"] == 875
    assert report["warnings"] == ["current_claim_supported_only_by_stale_evidence"]


def test_recent_evidence_avoids_stale_warning():
    report = analyze_evidence_claim_recency_mismatch(
        "The metric is latest available.",
        [{"id": "new", "content": "2026-04-20 release notes"}],
        now="2026-05-25",
    )

    assert report["warnings"] == []
    assert report["evidence"][0]["date"] == "2026-04-20"


def test_non_current_answer_and_empty_evidence_are_stable():
    report = analyze_evidence_claim_recency_mismatch("The 2020 report described growth.", [], now="2026-05-25")

    assert report == {"has_current_claim": False, "now": "2026-05-25", "evidence": [], "warnings": []}
