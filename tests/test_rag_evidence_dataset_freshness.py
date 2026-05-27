from __future__ import annotations

from datetime import date, datetime

from graph.rag.evidence_dataset_freshness import audit_evidence_dataset_freshness


def test_prefers_dataset_specific_dates_over_published_at():
    result = audit_evidence_dataset_freshness([{"id": "a", "dataset_date": "2025-12-01", "published_at": "2020-01-01"}], "2026-01-01")

    assert result["records"][0]["date_field"] == "dataset_date"
    assert result["bucket_counts"]["fresh"] == 1


def test_buckets_fresh_aging_stale_and_unknown():
    result = audit_evidence_dataset_freshness(
        [{"id": "f", "updated_at": "2025-10-01"}, {"id": "a", "updated_at": "2025-03-01"}, {"id": "s", "updated_at": "2024-01-01"}, {"id": "u"}],
        date(2026, 1, 1),
    )

    assert result["buckets"]["fresh"] == ["f"]
    assert result["buckets"]["aging"] == ["a"]
    assert result["buckets"]["stale"] == ["s"]
    assert result["buckets"]["unknown"] == ["u"]


def test_reference_date_accepts_datetime():
    result = audit_evidence_dataset_freshness([{"updated_at": "2026-01-01"}], datetime(2026, 1, 2))

    assert result["records"][0]["age_days"] == 1
