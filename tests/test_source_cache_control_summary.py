from __future__ import annotations

from graph.store import summarize_source_cache_controls


def test_cache_control_summary_groups_directives_and_preserves_examples():
    summary = summarize_source_cache_controls(
        [
            {"id": "b", "headers": {"Cache-Control": "No-Store, max-age=3600; private"}},
            {"id": "a", "metadata": {"response_headers": {"cache_control": "public, immutable"}}},
            {"id": "c", "cache-control": "no-cache, stale-while-revalidate=60"},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_cache_control"] == 3
    assert summary["missing_cache_control_count"] == 1
    assert summary["directive_counts"]["max-age"] == 1
    assert summary["directive_counts"]["no-store"] == 1
    assert summary["directive_counts"]["stale-while-revalidate"] == 1
    assert {"directive": "stale-while-revalidate", "count": 1, "source_ids": ["c"], "examples": ["stale-while-revalidate=60"]} in summary["rows"]


def test_cache_control_rows_are_sorted_and_sample_limited():
    summary = summarize_source_cache_controls(
        [{"id": "z", "cache_control": "private"}, {"id": "a", "cache_control": "private"}],
        sample_limit=1,
    )

    assert summary["rows"] == [{"directive": "private", "count": 2, "source_ids": ["z"], "examples": ["private"]}]
