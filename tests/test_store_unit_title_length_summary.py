from __future__ import annotations

from graph.store.unit_title_length_summary import summarize_unit_title_lengths


def test_title_length_summary_groups_by_source_and_tracks_lengths():
    summary = summarize_unit_title_lengths(
        [
            {"id": "a", "source_project": "s", "title": "A"},
            {"id": "b", "source_project": "s", "title": "Long title"},
            {"id": "c", "source_project": "s", "title": "  "},
            {"id": "d", "metadata": {"source": "other", "title": "Meta"}},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "other",
            "unit_count": 1,
            "missing_title_count": 0,
            "min_title_length": 4,
            "max_title_length": 4,
            "average_title_length": "4.00",
            "shortest_title_unit_id": "d",
            "longest_title_unit_id": "d",
        },
        {
            "source": "s",
            "unit_count": 3,
            "missing_title_count": 1,
            "min_title_length": 1,
            "max_title_length": 10,
            "average_title_length": "5.50",
            "shortest_title_unit_id": "a",
            "longest_title_unit_id": "b",
        },
    ]
