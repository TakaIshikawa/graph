import pytest

from graph.store import summarize_collection_stale_members


def test_collection_stale_members_counts_by_cutoff_with_timestamp_fallbacks():
    report = summarize_collection_stale_members(
        [
            {
                "id": "c1",
                "members": [
                    {"id": "old", "updated_at": "2025-01-01"},
                    {"id": "fresh", "modified_at": "2025-05-01"},
                    {"id": "bad", "created_at": "not-a-date"},
                ],
            },
            {"id": "c2", "metadata": {"units": [{"id": "created", "created_at": "2024-12-31"}]}},
        ],
        cutoff_date="2025-04-01",
    )

    assert report["collection_count"] == 2
    assert report["stale_collection_count"] == 2
    assert report["stale_member_count"] == 2
    assert report["invalid_timestamp_count"] == 1
    assert report["counts_by_collection"] == [
        {"collection_id": "c1", "stale_member_count": 1},
        {"collection_id": "c2", "stale_member_count": 1},
    ]
    assert report["samples"] == [
        {"collection_id": "c1", "member_id": "old", "timestamp": "2025-01-01"},
        {"collection_id": "c2", "member_id": "created", "timestamp": "2024-12-31"},
    ]
    assert report["invalid_timestamp_samples"] == [{"collection_id": "c1", "member_id": "bad", "timestamp": "not-a-date"}]


def test_collection_stale_members_supports_max_age_days_and_validates_inputs():
    report = summarize_collection_stale_members(
        [{"id": "c", "members": [{"id": "stale", "updated_at": "2025-01-01"}, {"id": "fresh", "updated_at": "2025-01-20"}]}],
        max_age_days=10,
        reference_date="2025-01-25",
    )

    assert report["stale_member_count"] == 1

    with pytest.raises(ValueError):
        summarize_collection_stale_members([])

    with pytest.raises(ValueError):
        summarize_collection_stale_members([], cutoff_date="2025-01-01", max_age_days=5)
