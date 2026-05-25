from __future__ import annotations

from graph.store.unit_link_rot_risk_summary import summarize_unit_link_rot_risk


def test_summarize_unit_link_rot_risk_buckets_clean_failing_archived_stale_and_missing():
    summary = summarize_unit_link_rot_risk(
        [
            {"id": "none", "metadata": {}},
            {"id": "clean", "metadata": {"links": [{"url": "https://ok.test", "status": 200}]}},
            {"id": "fail", "metadata": {"links": [{"url": "https://bad.test", "status_code": 500}]}},
            {"id": "archived", "metadata": {"links": [{"url": "https://gone.test", "archive_url": "https://archive.test/gone"}]}},
            {
                "id": "stale",
                "metadata": {"links": [{"url": "https://old.test", "checked_at": "2026-04-01T00:00:00+00:00"}]},
            },
        ],
        reference_date="2026-05-15T00:00:00+00:00",
        stale_after_days=30,
    )

    assert summary == {
        "total_units": 5,
        "urls": 4,
        "failing_urls": 1,
        "archived_urls": 1,
        "stale_checks": 1,
        "risk_buckets": {
            "no_links": 1,
            "clean": 1,
            "archived": 1,
            "stale": 1,
            "failing": 1,
        },
    }


def test_summarize_unit_link_rot_risk_failing_bucket_takes_precedence():
    summary = summarize_unit_link_rot_risk(
        [{"metadata": {"links": [{"status": 404, "archived": True, "stale": True}]}}],
        reference_date="2026-05-15T00:00:00+00:00",
    )

    assert summary["failing_urls"] == 1
    assert summary["archived_urls"] == 1
    assert summary["stale_checks"] == 1
    assert summary["risk_buckets"]["failing"] == 1
