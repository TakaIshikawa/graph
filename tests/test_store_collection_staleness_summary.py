from __future__ import annotations

from graph.store.collection_staleness_summary import collection_staleness_summary


class Collection:
    def __init__(self, collection_id: str, metadata: dict):
        self.id = collection_id
        self.metadata = metadata


def test_collection_staleness_summary_buckets_relative_to_reference_date():
    rows = collection_staleness_summary(
        [
            {"id": "fresh", "updated_at": "2026-05-01", "member_ids": ["a", "b"]},
            {"id": "aging", "metadata": {"modified_at": "2026-03-20", "members": ["a"]}},
            Collection("stale", {"last_seen_at": "2026-01-01", "unit_ids": ["a", "b", "c"]}),
            {"id": "unknown", "updated_at": "not-a-date"},
        ],
        reference_date="2026-05-27",
        stale_after_days=90,
    )

    assert rows == [
        {
            "collection_id": "aging",
            "age_days": 68,
            "staleness_bucket": "aging",
            "is_stale": False,
            "member_count": 1,
            "sample_collection_ids": ["aging"],
        },
        {
            "collection_id": "fresh",
            "age_days": 26,
            "staleness_bucket": "fresh",
            "is_stale": False,
            "member_count": 2,
            "sample_collection_ids": ["fresh"],
        },
        {
            "collection_id": "stale",
            "age_days": 146,
            "staleness_bucket": "stale",
            "is_stale": True,
            "member_count": 3,
            "sample_collection_ids": ["stale"],
        },
        {
            "collection_id": "unknown",
            "age_days": None,
            "staleness_bucket": "unknown",
            "is_stale": False,
            "member_count": 0,
            "sample_collection_ids": ["unknown"],
        },
    ]
