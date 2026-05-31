from __future__ import annotations

from datetime import datetime, timezone

from graph.store.source_auth_expiry_summary import summarize_source_auth_expiry


def test_source_auth_expiry_buckets_metadata_dates_against_reference_date():
    summary = summarize_source_auth_expiry(
        [
            {"id": "expired", "metadata": {"token_expires_at": "2026-05-30T00:00:00Z"}},
            {"id": "soon", "metadata": {"oauth_expires_at": "2026-06-14T00:00:00Z"}},
            {"id": "valid", "metadata": {"credential_expires_at": "2026-06-20T00:00:00Z"}},
            {"id": "missing", "metadata": {}},
            {"id": "invalid", "metadata": {"auth_expires_at": "next week"}},
        ],
        reference_date="2026-05-31T00:00:00Z",
    )

    assert summary["total_sources"] == 5
    assert summary["bucket_counts"] == {"expired": 1, "expiring_soon": 1, "valid": 1, "unknown": 2}
    assert summary["field_counts"] == {
        "metadata.auth_expires_at": 1,
        "metadata.credential_expires_at": 1,
        "metadata.oauth_expires_at": 1,
        "metadata.token_expires_at": 1,
    }
    assert summary["reference_date"] == "2026-05-31T00:00:00+00:00"
    assert summary["expiring_soon_days"] == 14


def test_source_auth_expiry_supports_datetime_reference_top_level_fields_and_sample_limit():
    summary = summarize_source_auth_expiry(
        [
            {"source_id": "b", "expires_at": "2026-06-01"},
            {"source_id": "a", "metadata": {"expires_at": "bad"}},
        ],
        reference_date=datetime(2026, 5, 31, tzinfo=timezone.utc),
        sample_limit=1,
    )

    assert summary["bucket_counts"] == {"expired": 0, "expiring_soon": 1, "valid": 0, "unknown": 1}
    assert summary["field_counts"] == {"expires_at": 1, "metadata.expires_at": 1}
    assert len(summary["samples"]) == 1
    assert summary["samples"][0]["bucket"] == "expiring_soon"


def test_source_auth_expiry_handles_empty_sources_and_negative_sample_limit():
    assert summarize_source_auth_expiry([], reference_date="2026-05-31", sample_limit=-1) == {
        "total_sources": 0,
        "bucket_counts": {"expired": 0, "expiring_soon": 0, "valid": 0, "unknown": 0},
        "field_counts": {},
        "reference_date": "2026-05-31T00:00:00+00:00",
        "expiring_soon_days": 14,
        "samples": [],
    }
