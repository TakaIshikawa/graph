from __future__ import annotations

from graph.store import summarize_source_redirect_chain_depths


def test_source_redirect_chain_depth_summary_derives_counts_and_domain_changes():
    summary = summarize_source_redirect_chain_depths(
        [
            {"id": "s1", "redirect_count": 2, "url": "https://a.test/x", "final_url": "https://b.test/y"},
            {"id": "s2", "metadata": {"redirect_chain": ["a", "b", "c"]}},
            {"id": "s3", "redirects": "a -> b"},
            {"id": "s4"},
        ],
        sample_limit=2,
    )

    assert summary == {
        "source_count": 4,
        "depth_buckets": {"2": 2, "3": 1},
        "longest_chains": [{"source_id": "s2", "redirect_depth": 3}, {"source_id": "s1", "redirect_depth": 2}],
        "final_domain_change_count": 1,
        "sources_missing_redirect_depth": ["s4"],
    }
