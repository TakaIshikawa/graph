from __future__ import annotations

from graph.store import summarize_unit_timeline_gaps


def test_timeline_gap_summary_reports_largest_gap_per_group():
    summary = summarize_unit_timeline_gaps(
        [
            {"id": "a", "metadata": {"collection": "c1", "source": "rss", "created_at": "2024-01-01"}},
            {"id": "b", "metadata": {"collection": "c1", "source": "rss", "published_at": "2024-01-10T00:00:00Z"}},
            {"id": "c", "metadata": {"collection": "c1", "source": "rss", "date": "2024-01-12"}},
            {"id": "d", "metadata": {"collection": "c2", "source": "rss", "captured_at": "bad"}},
        ]
    )

    assert summary["skipped_units"] == 1
    assert summary["rows"] == [
        {
            "collection": "c1",
            "source": "rss",
            "unit_count": 3,
            "first_timestamp": "2024-01-01",
            "last_timestamp": "2024-01-12",
            "largest_gap_days": 9,
            "gap_days": 9,
            "before_unit_id": "a",
            "after_unit_id": "b",
        }
    ]
