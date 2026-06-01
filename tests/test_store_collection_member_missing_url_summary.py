from __future__ import annotations

from graph.store import summarize_collection_member_missing_urls


def test_collection_member_missing_url_summary_counts_absent_blank_and_non_string_urls():
    summary = summarize_collection_member_missing_urls(
        [{"id": "c1", "member_ids": ["u1", "u2", "u3", "u4"]}, {"id": "c2", "members": [{"id": "m", "url": "https://ok.test"}]}],
        units=[{"id": "u1", "url": "https://ok.test"}, {"id": "u2", "url": ""}, {"id": "u3", "url": 123}, {"id": "u4"}],
        sample_limit=2,
    )

    assert summary == {
        "collection_count": 2,
        "affected_collection_count": 1,
        "collections": [
            {"collection_id": "c1", "member_count": 4, "missing_url_count": 3, "url_coverage_ratio": 0.25, "sample_member_ids": ["u2", "u3"]}
        ],
    }
