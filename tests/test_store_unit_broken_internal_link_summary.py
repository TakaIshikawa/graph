from __future__ import annotations

from graph.store import summarize_unit_broken_internal_links


def test_unit_broken_internal_link_summary_resolves_ids_slugs_and_titles():
    summary = summarize_unit_broken_internal_links(
        [
            {"id": "alpha", "title": "Alpha Note", "slug": "alpha-note", "content": "[ok](beta) [also ok](Gamma Note) [bad](missing-one)\n[[missing-two]]"},
            {"id": "beta", "title": "Beta"},
            {"id": "gamma", "title": "Gamma Note"},
        ]
    )

    assert summary["total_units"] == 3
    assert summary["internal_link_count"] == 4
    assert summary["broken_link_count"] == 2
    assert summary["missing_targets"] == [{"target": "missing-one", "count": 1}, {"target": "missing-two", "count": 1}]
    assert summary["source_unit_counts"] == [{"unit_id": "alpha", "count": 2}]


def test_unit_broken_internal_link_summary_ignores_external_urls_and_fences():
    summary = summarize_unit_broken_internal_links([{"id": "u", "content": "[external](https://example.com)\n```\n[bad](missing)\n```\n[#anchor](#local)"}])

    assert summary["internal_link_count"] == 0
    assert summary["broken_link_count"] == 0
