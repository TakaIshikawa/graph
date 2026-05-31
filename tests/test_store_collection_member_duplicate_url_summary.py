from __future__ import annotations

from graph.store.collection_member_duplicate_url_summary import summarize_collection_member_duplicate_urls


def test_collection_member_duplicate_urls_detects_duplicates_within_collection_only():
    summary = summarize_collection_member_duplicate_urls(
        [
            {"id": "c1", "members": [{"id": "m1", "url": "https://example.com/a"}, {"id": "m2", "url": "https://example.com/a#top"}]},
            {"id": "c2", "members": [{"id": "m3", "url": "https://example.com/a"}]},
        ]
    )

    assert summary["collections"] == [
        {"collection_id": "c1", "duplicate_url_count": 1, "examples": [{"url": "https://example.com/a", "member_ids": ["m1", "m2"]}]}
    ]


def test_collection_member_duplicate_urls_uses_unit_lookup_and_normalizes_trailing_slashes():
    summary = summarize_collection_member_duplicate_urls(
        [{"id": "c1", "member_ids": ["u1", "u2", "u3"]}],
        [{"id": "u1", "canonical_url": "https://example.com/a/"}, {"id": "u2", "metadata": {"source_url": "https://example.com/a#section"}}, {"id": "u3", "url": "https://example.com/b"}],
    )

    assert summary["collections"][0]["duplicate_url_count"] == 1
    assert summary["collections"][0]["examples"] == [{"url": "https://example.com/a", "member_ids": ["u1", "u2"]}]
