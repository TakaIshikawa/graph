from __future__ import annotations

from dataclasses import dataclass

from graph.store import summarize_unit_callout_usage


@dataclass
class Unit:
    content: str
    metadata: dict[str, str]


def test_summarize_unit_callout_usage_counts_case_insensitive_markers_and_folds():
    summary = summarize_unit_callout_usage(
        [
            {"source_project": "notes", "content": "> [!NOTE]+ Title\n> body\n> [!warning]- Risk"},
            {"source_project": "notes", "content": "> [!note] Again"},
            {"source_project": "notes", "content": "plain"},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "notes",
            "unit_count": 3,
            "units_with_callouts": 2,
            "callout_count": 3,
            "most_common_callout_type": "note",
            "folded_callout_count": 2,
            "max_callouts_per_unit": 2,
        }
    ]


def test_summarize_unit_callout_usage_supports_objects_sorted_sources_and_tie_breaks():
    summary = summarize_unit_callout_usage(
        [
            Unit(content="> [!tip] One\n> [!note] Two", metadata={"source": "Beta"}),
            {"metadata": {"source": "alpha"}, "content": "No callouts"},
        ]
    )

    assert [row["source"] for row in summary["rows"]] == ["alpha", "Beta"]
    assert summary["rows"][0]["most_common_callout_type"] == ""
    assert summary["rows"][1]["most_common_callout_type"] == "note"
