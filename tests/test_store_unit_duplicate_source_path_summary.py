from __future__ import annotations

from graph.store import summarize_unit_duplicate_source_paths


def test_duplicate_source_path_summary_groups_normalized_paths():
    summary = summarize_unit_duplicate_source_paths(
        [
            {"id": "a", "source_project": "md", "metadata": {"source_path": "notes/a.md"}},
            {"id": "b", "metadata": {"source_path": "notes/a.md/", "source_type": "markdown"}},
            {"id": "c", "metadata": {"source_path": "notes/b.md"}},
            {"id": "d", "metadata": {}},
        ]
    )

    assert summary["skipped_units"] == 1
    assert summary["duplicate_paths"] == [
        {"source_path": "notes/a.md", "unit_ids": ["a", "b"], "count": 2, "source_types": ["markdown", "md"]}
    ]
