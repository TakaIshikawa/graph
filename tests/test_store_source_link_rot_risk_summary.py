from datetime import datetime, timezone

from graph.store import summarize_source_link_rot_risks


def test_source_link_rot_risk_summary_buckets_errors_stale_and_archived():
    summary = summarize_source_link_rot_risks(
        [
            {"source_id": "ok", "status_code": 200, "last_checked_at": "2026-05-01T00:00:00Z"},
            {"source_id": "client", "status_code": 404},
            {"source_id": "server", "metadata": {"status_code": 503, "archive_url": "https://archive.test/x"}},
            {"source_id": "error", "fetch_error": "timeout"},
            {"source_id": "stale", "last_checked_at": "2025-01-01T00:00:00Z"},
            {"source_id": "invalid", "last_checked_at": "not a date"},
        ],
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert summary["risk_counts"] == {"client_error": 1, "fetch_error": 1, "ok": 1, "server_error": 1, "stale_check": 2}
    assert summary["high_risk_count"] == 3
    assert summary["archived_count"] == 1
    assert summary["stale_check_count"] == 2
    assert any(sample["source_id"] == "invalid" and sample["invalid_date"] for sample in summary["samples"])
