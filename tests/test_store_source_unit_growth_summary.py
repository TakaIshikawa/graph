from __future__ import annotations

from graph.store.source_unit_growth_summary import source_unit_growth_summary


def test_source_unit_growth_groups_months_and_cumulative_counts():
    rows = source_unit_growth_summary(
        [
            {"source_project": "pocket", "created_at": "2024-01-02T10:00:00Z"},
            {"source_project": "pocket", "ingested_at": "2024-01-10T10:00:00+00:00"},
            {"source_project": "pocket", "created_at": "2024-02-01"},
        ]
    )

    assert rows == [
        {
            "source_project": "pocket",
            "month": "2024-01",
            "unit_count": 2,
            "cumulative_count": 2,
            "first_timestamp": "2024-01-02T10:00:00+00:00",
            "latest_timestamp": "2024-01-10T10:00:00+00:00",
        },
        {
            "source_project": "pocket",
            "month": "2024-02",
            "unit_count": 1,
            "cumulative_count": 3,
            "first_timestamp": "2024-02-01T00:00:00+00:00",
            "latest_timestamp": "2024-02-01T00:00:00+00:00",
        },
    ]


def test_source_unit_growth_invalid_timestamps_use_bucket():
    rows = source_unit_growth_summary([{"source_project": "x", "created_at": "bad"}])

    assert rows[0]["month"] == "invalid_timestamp"
    assert rows[0]["first_timestamp"] == ""
