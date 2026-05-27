from __future__ import annotations

from datetime import datetime, timezone

from graph.store.unit_source_lag_summary import unit_source_lag_summary


class Unit:
    def __init__(self, id: str, source_project: str, metadata: dict, ingested_at: str):
        self.id = id
        self.source_project = source_project
        self.metadata = metadata
        self.ingested_at = ingested_at


def test_unit_source_lag_summary_computes_lag_by_source():
    rows = unit_source_lag_summary(
        [
            Unit(
                "u1",
                "web",
                {"source_updated_at": "2026-05-01T00:00:00Z"},
                "2026-05-01T06:00:00Z",
            ),
            Unit(
                "u2",
                "web",
                {"published_at": "2026-05-01T00:00:00+00:00"},
                "2026-05-01T12:00:00+00:00",
            ),
        ]
    )

    assert rows == [
        {
            "source": "web",
            "count": 2,
            "average_lag_hours": 9.0,
            "max_lag_hours": 12.0,
            "negative_lag_count": 0,
            "missing_timestamp_count": 0,
            "sample_unit_ids": ["u1", "u2"],
        }
    ]


def test_unit_source_lag_summary_counts_negative_and_missing_timestamps():
    rows = unit_source_lag_summary(
        [
            {
                "id": "u1",
                "source": "api",
                "source_updated_at": "2026-05-02T00:00:00Z",
                "ingested_at": "2026-05-01T00:00:00Z",
                "metadata": {},
            },
            {
                "id": "u2",
                "source": "api",
                "metadata": {"source_created_at": "not a date"},
                "created_at": "2026-05-01T00:00:00Z",
            },
            {
                "id": "u3",
                "metadata": {
                    "source": "api",
                    "published_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
                },
                "ingested_at": datetime(2026, 5, 1, 3, tzinfo=timezone.utc),
            },
        ]
    )

    assert rows == [
        {
            "source": "api",
            "count": 3,
            "average_lag_hours": 3.0,
            "max_lag_hours": 3.0,
            "negative_lag_count": 1,
            "missing_timestamp_count": 1,
            "sample_unit_ids": ["u1", "u2", "u3"],
        }
    ]


def test_unit_source_lag_summary_bounds_sample_unit_ids():
    rows = unit_source_lag_summary(
        [
            {
                "id": f"u{i}",
                "source_project": "web",
                "source_updated_at": "2026-05-01T00:00:00Z",
                "ingested_at": "2026-05-01T01:00:00Z",
                "metadata": {},
            }
            for i in range(4)
        ],
        sample_limit=2,
    )

    assert rows[0]["sample_unit_ids"] == ["u0", "u1"]
