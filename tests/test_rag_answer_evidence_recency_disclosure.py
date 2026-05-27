from __future__ import annotations

from graph.rag.answer_evidence_recency_disclosure import audit_answer_evidence_recency_disclosure


def test_evidence_recency_disclosure_flags_missing_stale_disclosure():
    report = audit_answer_evidence_recency_disclosure(
        "The evidence supports adoption.",
        [{"id": "old", "date": "2018-01-01"}, {"id": "new", "date": "2025-01-01"}],
        now="2026-01-01",
    )

    assert report["needs_recency_disclosure"] is True
    assert report["passes"] is False
    assert report["has_mixed_age_evidence"] is True


def test_evidence_recency_disclosure_passes_when_answer_acknowledges_range():
    report = audit_answer_evidence_recency_disclosure(
        "Evidence spans from 2018 through 2025, so older evidence may be stale.",
        [{"date": "2018-01-01"}, {"date": "2025-01-01"}],
        now="2026-01-01",
    )

    assert report["has_recency_disclosure"] is True
    assert report["passes"] is True


def test_evidence_recency_disclosure_handles_missing_dates():
    report = audit_answer_evidence_recency_disclosure("No date caveat.", [{"id": "x"}, {"date": "2025-02-01"}], now="2026-01-01")

    assert report["date_count"] == 1
    assert report["missing_date_count"] == 1
    assert report["passes"] is True
