from __future__ import annotations

from graph.store.unit_recurring_title_pattern_summary import summarize_unit_recurring_title_patterns


def test_unit_recurring_title_patterns_normalizes_dates_numbers_and_uuids():
    summary = summarize_unit_recurring_title_patterns(
        [
            {"source_project": "notes", "title": "Daily note 2024-01-01"},
            {"source_project": "notes", "title": "Daily note 2024-01-02"},
            {"source_project": "notes", "title": "Invoice 123 total 45.67"},
            {"source_project": "notes", "title": "Invoice 456 total 89.10"},
            {"source_project": "web", "title": "Run 550e8400-e29b-41d4-a716-446655440000"},
            {"source_project": "web", "title": "Run 550e8400-e29b-41d4-a716-446655440001"},
            {"source_project": "web", "title": "Only once 1"},
        ]
    )

    assert summary["total_units"] == 7
    assert summary["recurring_pattern_count"] == 3
    assert [row["pattern"] for row in summary["pattern_summaries"]] == [
        "Daily note {date}",
        "Invoice {number} total {number}",
        "Run {uuid}",
    ]


def test_unit_recurring_title_patterns_orders_sources_and_excludes_singletons():
    summary = summarize_unit_recurring_title_patterns(
        [
            {"source_project": "b", "title": "Report 1"},
            {"source_project": "a", "title": "Report 2"},
            {"source_project": "a", "title": "Report 3"},
            {"source_project": "b", "title": "Unique 1"},
        ]
    )

    assert summary["pattern_summaries"] == [
        {"source": "a", "source_project": "a", "pattern": "Report {number}", "unit_count": 2, "sample_titles": ["Report 2", "Report 3"]}
    ]
