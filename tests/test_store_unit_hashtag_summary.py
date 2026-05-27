from __future__ import annotations

from graph.store import summarize_unit_hashtags


def test_hashtag_summary_counts_nested_tags_and_ignores_urls_and_code():
    report = summarize_unit_hashtags([{"id": "u", "content": "#Tag #tag/a/b https://x.test/#frag\n```\n#ignored\n```\n#tag"}])

    assert report["total_occurrences"] == 3
    assert report["unique_normalized_tags"] == 2
    assert report["nested_tag_depth_distribution"] == [{"depth": 1, "count": 2}, {"depth": 3, "count": 1}]
    assert report["top_tags"][0]["tag"] == "tag"
