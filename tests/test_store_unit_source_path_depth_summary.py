from __future__ import annotations

from graph.store.unit_source_path_depth_summary import summarize_unit_source_path_depth


def test_unit_source_path_depth_counts_filesystem_and_url_paths():
    summary = summarize_unit_source_path_depth(
        [
            {"source_project": "docs", "metadata": {"source_path": "readme.md"}},
            {"source_project": "docs", "metadata": {"file_path": "/a/b/c/d.txt"}},
            {"source_project": "docs", "metadata": {"url_path": "https://example.test/a/b/c"}},
            {"source_project": "docs", "metadata": {}},
            {"source_project": "web", "path": "https://example.test/"},
        ]
    )

    rows = {row["source"]: row for row in summary["rows"]}
    assert rows["docs"]["path_count"] == 3
    assert rows["docs"]["missing_path_count"] == 1
    assert rows["docs"]["min_depth"] == 1
    assert rows["docs"]["max_depth"] == 4
    assert rows["docs"]["average_depth"] == 2.67
    assert rows["docs"]["root_level_count"] == 1
    assert rows["docs"]["deep_path_count"] == 1
    assert rows["web"]["min_depth"] == 0


def test_unit_source_path_depth_orders_sources_and_ignores_missing_for_min_max():
    summary = summarize_unit_source_path_depth([{"source_project": "b"}, {"source_project": "a", "path": "one/two"}])

    assert [row["source"] for row in summary["rows"]] == ["a", "b"]
    assert summary["rows"][1]["min_depth"] == 0
