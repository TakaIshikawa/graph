from __future__ import annotations

from graph.store.source_robots_policy_summary import summarize_source_robots_policies


def test_source_robots_policy_normalizes_booleans_and_strings():
    summary = summarize_source_robots_policies(
        [
            {"id": "a", "metadata": {"robots_allowed": True}},
            {"id": "b", "crawl_allowed": False},
            {"id": "c", "metadata": {"robots_policy": "allowed"}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["policy_counts"] == {"allowed": 2, "disallowed": 1}
    assert summary["allowed_count"] == 2
    assert summary["disallowed_count"] == 1
    assert summary["noindex_count"] == 0
    assert summary["missing_policy_count"] == 1


def test_source_robots_policy_counts_x_robots_noindex_and_limits_samples():
    summary = summarize_source_robots_policies(
        [
            {"source_id": "a", "metadata": {"x_robots_tag": "noindex, nofollow"}},
            {"source_id": "b", "noindex": True},
        ],
        sample_limit=1,
    )

    assert summary["policy_counts"] == {"noindex": 2}
    assert summary["disallowed_count"] == 2
    assert summary["noindex_count"] == 2
    assert summary["samples"] == [
        {"source_id": "a", "field": "metadata.x_robots_tag", "policy": "noindex", "value": "noindex, nofollow"}
    ]
