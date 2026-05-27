from __future__ import annotations

from graph.store import summarize_unit_duplicate_titles


def test_duplicate_title_summary_groups_normalized_titles():
    report = summarize_unit_duplicate_titles([{"id": "b", "title": " Alpha "}, {"id": "a", "title": "alpha"}, {"id": "c", "title": "Beta"}])

    assert report["duplicate_group_count"] == 1
    assert report["duplicate_groups"] == [{"normalized_title": "alpha", "duplicate_count": 2, "unit_ids": ["a", "b"], "sample_titles": ["Alpha", "alpha"]}]


def test_duplicate_title_summary_omits_unique_titles():
    assert summarize_unit_duplicate_titles([{"id": "a", "title": "One"}, {"id": "b", "title": "Two"}])["duplicate_groups"] == []


def test_duplicate_title_summary_ignores_missing_titles():
    report = summarize_unit_duplicate_titles([{"id": "a"}, {"id": "b", "title": ""}, {"id": "c", "title": " X "}, {"id": "d", "title": "x"}])

    assert report["duplicate_groups"][0]["unit_ids"] == ["c", "d"]
