from __future__ import annotations

from graph.store.collection_update_sla_summary import summarize_collection_update_slas


def test_summarize_collection_update_slas_uses_reference_date_and_owner_groups():
    summary = summarize_collection_update_slas(
        [
            {"name": "current", "owner": "docs", "sla_days": 10, "last_updated_at": "2026-05-20T00:00:00+00:00"},
            {"name": "stale", "owner": "api", "sla_days": 7, "last_updated_at": "2026-05-01T00:00:00+00:00"},
            {"name": "missing", "metadata": {"owner": "docs", "update_sla_days": 14}},
            {"name": "unconfigured", "owner": "api", "last_updated_at": "2026-01-01T00:00:00+00:00"},
        ],
        reference_date="2026-05-26T00:00:00+00:00",
    )

    assert summary == {
        "total_collections": 4,
        "configured_collections": 3,
        "breached_collections": 2,
        "missing_last_updated_count": 1,
        "average_breach_days": 18.0,
        "breach_by_owner": [{"owner": "api", "count": 1}, {"owner": "docs", "count": 1}],
    }


def test_summarize_collection_update_slas_counts_unconfigured_without_breach():
    summary = summarize_collection_update_slas(
        [{"name": "unconfigured", "last_updated_at": "2026-01-01T00:00:00+00:00"}],
        reference_date="2026-05-26T00:00:00+00:00",
    )

    assert summary["configured_collections"] == 0
    assert summary["breached_collections"] == 0
