from __future__ import annotations

from graph.store import summarize_unit_frontmatter_required_keys


def test_required_key_summary_counts_complete_metadata():
    report = summarize_unit_frontmatter_required_keys([{"id": "a", "metadata": {"title": "T", "date": "D"}}], ["title", "date"])

    assert report["complete_units"] == 1
    assert report["incomplete_units"] == 0


def test_required_key_summary_counts_partial_metadata_deterministically():
    report = summarize_unit_frontmatter_required_keys([{"id": "b", "metadata": {"title": "T"}}, {"id": "a", "metadata": {}}], ["title", "date"])

    assert report["missing_key_counts"] == [{"key": "date", "count": 2}, {"key": "title", "count": 1}]
    assert report["examples_by_missing_key"] == {"date": ["b", "a"], "title": ["a"]}


def test_required_key_summary_empty_required_keys_treats_all_complete():
    report = summarize_unit_frontmatter_required_keys([{"id": "a"}, {"id": "b", "metadata": {"x": "y"}}], [])

    assert report["complete_units"] == 2
    assert report["missing_key_counts"] == []
