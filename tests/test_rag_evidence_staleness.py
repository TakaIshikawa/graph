import pytest

from graph.rag.evidence_staleness import audit_evidence_staleness


def test_evidence_staleness_counts_stale_undated_and_invalid_dates():
    report = audit_evidence_staleness(
        [
            {"id": "old", "published_at": "2024-01-01"},
            {"id": "fresh", "updated_at": "2025-05-01"},
            {"id": "bad", "date": "not-a-date"},
            {"id": "none"},
        ],
        as_of="2025-05-01",
        stale_after_days=365,
    )

    assert report["evidence_count"] == 4
    assert report["dated_evidence_count"] == 2
    assert report["stale_evidence_count"] == 1
    assert report["undated_evidence_count"] == 2
    assert report["stale_ratio"] == 0.25
    assert [finding["type"] for finding in report["findings"]] == ["stale_evidence", "invalid_date", "undated_evidence"]


def test_evidence_staleness_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        audit_evidence_staleness([], stale_after_days=0)
