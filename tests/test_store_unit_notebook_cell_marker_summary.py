from __future__ import annotations

from graph.store import summarize_unit_notebook_cell_markers


def test_notebook_cell_marker_summary_detects_cell_variants():
    report = summarize_unit_notebook_cell_markers([{"id": "a", "content": "# %%\n# %% [markdown]\n// %%"}])

    assert report["total_markers"] == 3
    assert report["marker_type_counts"] == {"cell": 3}
    assert report["language_prefix_counts"] == {"#": 2, "//": 1}
    assert report["markdown_cell_count"] == 1


def test_notebook_cell_marker_summary_detects_region_balance():
    report = summarize_unit_notebook_cell_markers([{"id": "a", "content": "<!-- #region -->\ntext"}, {"id": "b", "content": "<!-- #region -->\n<!-- #endregion -->"}])

    assert report["unbalanced_region_units"] == 1
    assert report["marker_type_counts"] == {"endregion": 1, "region": 2}


def test_notebook_cell_marker_summary_examples_are_deterministic():
    report = summarize_unit_notebook_cell_markers([{"id": "a", "content": "# %%"}])

    assert report["examples"] == [{"unit_id": "a", "line": 1, "marker_type": "cell", "text": "# %%"}]
