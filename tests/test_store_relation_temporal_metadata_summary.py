from __future__ import annotations

from datetime import date, datetime

from graph.store import summarize_relation_temporal_metadata


class Relation:
    def __init__(self, relation_id: str, metadata: dict[str, object] | None = None, **values: object):
        self.id = relation_id
        self.metadata = metadata or {}
        for key, value in values.items():
            setattr(self, key, value)


def test_relation_temporal_metadata_summary_counts_iso_dates_and_datetimes():
    summary = summarize_relation_temporal_metadata(
        [
            {"id": "r1", "metadata": {"created_at": "2024-01-02", "updated_at": "2024-01-02T03:04:05Z"}},
            Relation("r2", metadata={"observed_at": date(2024, 2, 3)}, start_date=datetime(2024, 2, 3, 4, 5, 6)),
        ]
    )

    assert summary["relation_count"] == 2
    assert summary["key_presence_counts"] == {"created_at": 1, "observed_at": 1, "start_date": 1, "updated_at": 1}
    assert summary["valid_date_counts"] == {"created_at": 1, "observed_at": 1}
    assert summary["valid_datetime_counts"] == {"start_date": 1, "updated_at": 1}
    assert summary["invalid_temporal_values"] == []


def test_relation_temporal_metadata_summary_handles_missing_and_invalid_values():
    summary = summarize_relation_temporal_metadata(
        [
            {"id": "r1", "metadata": {}},
            {"id": "r2", "metadata": {"end_date": "next Tuesday", "updated_at": ""}},
            Relation("r3", created_at="not-a-date"),
        ]
    )

    assert summary["relation_count"] == 3
    assert summary["key_presence_counts"] == {"created_at": 1, "end_date": 1}
    assert summary["valid_date_counts"] == {}
    assert summary["valid_datetime_counts"] == {}
    assert summary["invalid_temporal_values"] == [
        {"relation_id": "r2", "key": "end_date", "value": "next Tuesday"},
        {"relation_id": "r3", "key": "created_at", "value": "not-a-date"},
    ]
