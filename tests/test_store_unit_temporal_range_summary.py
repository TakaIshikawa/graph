from __future__ import annotations

from graph.store import summarize_unit_temporal_ranges


def test_temporal_range_summary_reports_ranges_invalids_and_ties():
    summary = summarize_unit_temporal_ranges(
        [
            {"id": "b", "metadata": {"created_at": "2024-01-01T09:00:00+09:00", "date": "bad"}},
            {"id": "a", "metadata": {"created_at": "2023-12-31T23:00:00Z", "date": "2024-02-01"}},
            {"id": "c", "metadata": {"created_at": "2024-01-03", "date": "2024-02-03"}},
        ]
    )

    rows = {row["key"]: row for row in summary["rows"]}
    assert rows["created_at"]["parsed_count"] == 3
    assert rows["created_at"]["earliest_unit_id"] == "a"
    assert rows["created_at"]["latest_unit_id"] == "c"
    assert rows["created_at"]["span_days"] == 2
    assert rows["date"]["invalid_count"] == 1
