from __future__ import annotations

from graph.store.unit_content_freshness_summary import unit_content_freshness_summary


def test_unit_content_freshness_counts_dates_and_staleness():
    rows = unit_content_freshness_summary(
        [
            {"source_project": "notes", "source_entity_type": "note", "created_at": "2024-01-01", "updated_at": "2024-01-10"},
            {"source_project": "notes", "source_entity_type": "note", "created_at": "bad", "updated_at": "2024-03-01"},
            {"source_project": "notes", "source_entity_type": "note", "metadata": {"created": "2024-01-02", "updated": "bad"}},
        ],
        reference_date="2024-02-15",
        stale_after_days=30,
    )

    assert rows == [
        {
            "source_project": "notes",
            "source_entity_type": "note",
            "unit_count": 3,
            "missing_created_at_count": 1,
            "missing_updated_at_count": 1,
            "future_updated_at_count": 1,
            "stale_unit_count": 1,
            "latest_updated_at": "2024-03-01",
        }
    ]


def test_unit_content_freshness_sorts_groups():
    rows = unit_content_freshness_summary(
        [
            {"source_project": "b", "source_entity_type": "note"},
            {"source_project": "a", "source_entity_type": "article"},
        ]
    )

    assert [(row["source_project"], row["source_entity_type"]) for row in rows] == [("a", "article"), ("b", "note")]
