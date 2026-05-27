from __future__ import annotations

from graph.store.relation_temporal_coverage_summary import relation_temporal_coverage_summary


class Relation:
    def __init__(self, relation_id: str, metadata: dict):
        self.id = relation_id
        self.metadata = metadata


def test_relation_temporal_coverage_summary_groups_type_and_counts_coverage():
    rows = relation_temporal_coverage_summary(
        [
            {"id": "r1", "relation_type": "cites", "created_at": "2026-01-02T03:00:00Z"},
            {"id": "r2", "metadata": {"relation": "cites", "observed_at": "2026-01-05"}},
            {"id": "r3", "relation_type": "cites"},
            Relation("r4", {"type": "mentions", "start_date": "bad-date"}),
            Relation("r5", {"type": "mentions", "end_date": "2025-12-31"}),
        ]
    )

    assert rows == [
        {
            "relation_type": "cites",
            "total_count": 3,
            "dated_count": 2,
            "missing_date_count": 1,
            "coverage_share": 0.6667,
            "earliest_date": "2026-01-02",
            "latest_date": "2026-01-05",
            "sample_relation_ids": ["r1", "r2", "r3"],
        },
        {
            "relation_type": "mentions",
            "total_count": 2,
            "dated_count": 1,
            "missing_date_count": 1,
            "coverage_share": 0.5,
            "earliest_date": "2025-12-31",
            "latest_date": "2025-12-31",
            "sample_relation_ids": ["r4", "r5"],
        },
    ]


def test_relation_temporal_coverage_summary_uses_unknown_fallback():
    rows = relation_temporal_coverage_summary([{"id": "r1", "metadata": {"updated_at": "2026-02-01"}}, {"id": "r2"}])

    assert rows == [
        {
            "relation_type": "unknown",
            "total_count": 2,
            "dated_count": 1,
            "missing_date_count": 1,
            "coverage_share": 0.5,
            "earliest_date": "2026-02-01",
            "latest_date": "2026-02-01",
            "sample_relation_ids": ["r1", "r2"],
        }
    ]
