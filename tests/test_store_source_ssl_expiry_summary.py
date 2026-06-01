from __future__ import annotations

from datetime import datetime, timezone

from graph.store.source_ssl_expiry_summary import summarize_source_ssl_expiry


def test_source_ssl_expiry_buckets_iso_dates_and_datetimes():
    summary = summarize_source_ssl_expiry(
        [
            {"id": "expired", "metadata": {"ssl_expires_at": "2026-05-31"}},
            {"id": "soon", "tls_expires_at": "2026-06-30T00:00:00Z"},
            {"id": "valid", "metadata": {"certificate_expires_at": "2026-07-15T00:00:00+00:00"}},
            {"id": "missing"},
        ],
        now="2026-06-01T00:00:00Z",
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_ssl_expiry"] == 3
    assert summary["expired_count"] == 1
    assert summary["expiring_soon_count"] == 1
    assert summary["missing_ssl_expiry_count"] == 1
    assert summary["bucket_counts"] == {"expired": 1, "expiring_soon": 1, "valid": 1, "missing": 1, "invalid": 0}


def test_source_ssl_expiry_reports_invalid_dates_without_raising_and_limits_samples():
    summary = summarize_source_ssl_expiry(
        [
            {"source_id": "invalid", "metadata": {"cert_expires_at": "next week"}},
            {"source_id": "soon", "ssl_expires_at": "2026-06-15"},
        ],
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
        sample_limit=1,
    )

    assert summary["bucket_counts"] == {"expired": 0, "expiring_soon": 1, "valid": 0, "missing": 0, "invalid": 1}
    assert summary["samples"] == [
        {
            "source_id": "invalid",
            "field": "metadata.cert_expires_at",
            "ssl_expires_at": "next week",
            "bucket": "invalid",
            "days_until_expiry": None,
        }
    ]
