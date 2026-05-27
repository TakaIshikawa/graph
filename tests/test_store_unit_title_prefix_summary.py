from __future__ import annotations

from graph.store.unit_title_prefix_summary import summarize_unit_title_prefixes


def test_unit_title_prefix_summary_extracts_repeated_prefixes():
    summary = summarize_unit_title_prefixes(
        [
            {"id": "u2", "title": "[ADR] Storage", "source": "docs"},
            {"id": "u1", "title": "[ADR] API", "source": "docs"},
            {"id": "u3", "title": "2024-01-01 Notes", "source": "journal"},
            {"id": "u4", "title": "2024-01-01 Plan", "source": "journal"},
            {"id": "u5", "title": "Project: One"},
            {"id": "u6", "title": "Project: Two"},
            {"id": "u7", "title": "Area/Sub"},
            {"id": "u8", "title": "TODO item"},
        ]
    )

    assert [row["prefix"] for row in summary["prefix_counts"]] == ["2024-01-01", "[adr]", "project"]
    assert summary["prefix_counts"][1]["example_unit_ids"] == ["u1", "u2"]


def test_unit_title_prefix_summary_ignores_one_off_below_min_count():
    assert summarize_unit_title_prefixes([{"id": "u1", "title": "Only: Once"}])["prefix_counts"] == []
