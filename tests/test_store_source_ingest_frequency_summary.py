from __future__ import annotations

from datetime import datetime, timezone

from graph.store.source_ingest_frequency_summary import summarize_source_ingest_frequency


class Unit:
    def __init__(self, source_project: str, metadata: dict, created_at: object = None):
        self.source_project = source_project
        self.metadata = metadata
        self.created_at = created_at


def test_summarize_source_ingest_frequency_normalizes_iso_dates_by_source():
    summary = summarize_source_ingest_frequency(
        [
            {"source": "web", "ingested_at": "2026-05-01T10:15:00+00:00"},
            {"source": "web", "imported_at": "2026-05-01"},
            {"source": "web", "metadata": {"created_at": "2026-05-03T08:00:00+00:00"}},
        ]
    )

    assert summary == {
        "total_units": 3,
        "sources": [
            {
                "source": "web",
                "unit_count": 3,
                "dated_count": 3,
                "undated_count": 0,
                "first_ingest_date": "2026-05-01",
                "last_ingest_date": "2026-05-03",
                "active_day_count": 2,
                "average_units_per_active_day": 1.5,
            }
        ],
    }


def test_summarize_source_ingest_frequency_handles_z_timezone_values():
    summary = summarize_source_ingest_frequency(
        [
            {"source": "api", "ingested_at": "2026-05-02T23:30:00Z"},
            {"source": "api", "ingested_at": "2026-05-03T00:30:00+01:00"},
        ]
    )

    assert summary["sources"][0]["first_ingest_date"] == "2026-05-02"
    assert summary["sources"][0]["last_ingest_date"] == "2026-05-02"
    assert summary["sources"][0]["active_day_count"] == 1
    assert summary["sources"][0]["average_units_per_active_day"] == 2.0


def test_summarize_source_ingest_frequency_counts_invalid_and_missing_as_undated():
    summary = summarize_source_ingest_frequency(
        [
            {"source": "api", "ingested_at": "not-a-date"},
            {"source": "api", "metadata": {"imported_at": ""}},
            Unit("api", {}, datetime(2026, 5, 4, tzinfo=timezone.utc)),
        ]
    )

    assert summary["sources"] == [
        {
            "source": "api",
            "unit_count": 3,
            "dated_count": 1,
            "undated_count": 2,
            "first_ingest_date": "2026-05-04",
            "last_ingest_date": "2026-05-04",
            "active_day_count": 1,
            "average_units_per_active_day": 1.0,
        }
    ]


def test_summarize_source_ingest_frequency_orders_sources_deterministically():
    summary = summarize_source_ingest_frequency(
        [
            {"source": "zeta", "created_at": "2026-05-01T00:00:00Z"},
            {"metadata": {"source": "Alpha", "ingested_at": "2026-05-01T00:00:00Z"}},
            {"created_at": "2026-05-01T00:00:00Z"},
        ]
    )

    assert [row["source"] for row in summary["sources"]] == ["Alpha", "unknown", "zeta"]
