from datetime import datetime, timezone

from graph.store import summarize_source_expires_headers


def test_expires_summary_classifies_valid_invalid_and_missing_headers():
    summary = summarize_source_expires_headers(
        [
            {"source_id": "a", "Expires": "Tue, 01 Jan 2030 00:00:00 GMT"},
            {"source_id": "b", "metadata": {"headers": {"EXPIRES": "Tue, 01 Jan 2020 00:00:00 GMT"}}},
            {"source_id": "c", "expires": "not a date"},
            {"source_id": "d"},
        ],
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sample_limit=1,
    )

    assert summary["sources_with_expires"] == 3
    assert summary["sources_missing_expires"] == 1
    assert summary["expired_count"] == 1
    assert summary["future_expiry_count"] == 1
    assert summary["top_expiry_dates"] == {"2020-01-01": 1, "2030-01-01": 1}
    assert summary["invalid_expires_samples"] == [{"source_id": "c", "value": "not a date"}]
