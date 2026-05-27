from __future__ import annotations

from graph.store.unit_annotation_density_summary import summarize_unit_annotation_density


def test_unit_annotation_density_counts_mixed_shapes():
    summary = summarize_unit_annotation_density(
        [
            {"source_project": "notes", "metadata": {"annotations": ["a", "b"], "comments": "review"}},
            {"source_project": "notes", "metadata": {"highlights": 3, "notes": ["n"]}},
            {"source_project": "notes", "metadata": {"marginalia": None, "comments": -2}},
            {"source_project": "web", "comments": {"a": "x", "b": "y"}},
        ]
    )

    rows = {row["source"]: row for row in summary["rows"]}
    assert rows["notes"]["unit_count"] == 3
    assert rows["notes"]["annotated_count"] == 2
    assert rows["notes"]["comment_count"] == 1
    assert rows["notes"]["highlight_count"] == 3
    assert rows["notes"]["note_count"] == 1
    assert rows["notes"]["max_annotations"] == 4
    assert rows["notes"]["average_annotations"] == 2.33
    assert rows["web"]["comment_count"] == 2


def test_unit_annotation_density_missing_values_are_zero():
    summary = summarize_unit_annotation_density([{"metadata": {"source": "x", "annotations": ""}}, {"metadata": {}}])

    assert [row["source"] for row in summary["rows"]] == ["unknown", "x"]
    assert summary["rows"][1]["annotated_count"] == 0
