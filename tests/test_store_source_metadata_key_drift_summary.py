from __future__ import annotations

from graph.store.source_metadata_key_drift_summary import summarize_source_metadata_key_drift


def test_source_metadata_key_drift_reports_added_removed_and_stable_keys():
    summary = summarize_source_metadata_key_drift(
        [
            {"metadata": {"source": "s1", "created_at": "2024-01-01", "alpha": 1, "old": 1}},
            {"metadata": {"source": "s1", "created_at": "2024-02-01", "alpha": 2, "new": 1}},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "s1",
            "earlier_unit_count": 1,
            "later_unit_count": 1,
            "added_keys": ["new"],
            "removed_keys": ["old"],
            "stable_keys": ["alpha", "created_at", "source"],
        }
    ]


def test_source_metadata_key_drift_buckets_invalid_timestamps_without_crashing():
    summary = summarize_source_metadata_key_drift([{"source": "s2", "metadata": {"bad": 1, "updated_at": "bad-date"}}])

    assert summary["invalid_timestamp_count"] == 1
    assert summary["rows"][0]["added_keys"] == ["bad", "updated_at"]
