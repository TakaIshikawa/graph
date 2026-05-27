from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_author_coverage_summary import summarize_unit_author_coverage


def test_unit_author_coverage_counts_mapping_and_object_units():
    summary = summarize_unit_author_coverage(
        [
            {"source_project": "docs", "metadata": {"author": " Ada  Lovelace "}},
            {"source_project": "docs", "metadata": {"authors": ["Ada Lovelace", "Grace Hopper"]}},
            {"source_project": "docs", "metadata": {}},
            SimpleNamespace(source_project="notes", metadata={"creator": "Grace Hopper; Alan Kay"}),
            SimpleNamespace(source_project="notes", owner="Grace Hopper"),
        ]
    )

    assert summary["total_units"] == 5
    assert [row["source"] for row in summary["rows"]] == ["docs", "notes"]
    docs = summary["rows"][0]
    assert docs["unit_count"] == 3
    assert docs["authored_count"] == 2
    assert docs["missing_author_count"] == 1
    assert docs["multi_author_count"] == 1
    assert docs["top_authors"] == [{"author": "Ada Lovelace", "count": 2}, {"author": "Grace Hopper", "count": 1}]


def test_unit_author_coverage_normalizes_strings_and_orders_authors():
    summary = summarize_unit_author_coverage(
        [
            {"source_project": "z", "metadata": {"created_by": " bob , Alice "}},
            {"source_project": "a", "metadata": {"owner": ("Alice", "Bob")}},
            {"metadata": {"author": ""}},
        ]
    )

    assert [row["source"] for row in summary["source_summaries"]] == ["a", "unknown", "z"]
    assert summary["rows"][0]["top_authors"] == [{"author": "Alice", "count": 1}, {"author": "Bob", "count": 1}]
    assert summary["rows"][2]["multi_author_count"] == 1
