from __future__ import annotations

from graph.store.source_canonical_url_conflict_summary import summarize_source_canonical_url_conflicts


def test_source_canonical_url_conflicts_groups_many_sources_to_one_canonical():
    summary = summarize_source_canonical_url_conflicts(
        [
            {"id": "a", "url": "HTTP://Example.com/a/", "metadata": {"canonical_url": "https://example.com/canonical"}},
            {"id": "b", "final_url": "https://example.com/b", "canonical_url": "https://example.com/canonical/"},
            {"id": "c", "url": "https://example.com/c", "canonical_url": "https://example.com/c"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_canonical_url"] == 3
    assert summary["canonical_conflict_count"] == 1
    assert summary["url_conflict_count"] == 0
    assert summary["conflict_groups"] == [
        {
            "type": "canonical",
            "canonical_url": "https://example.com/canonical",
            "urls": ["http://example.com/a", "https://example.com/b"],
            "count": 2,
        }
    ]


def test_source_canonical_url_conflicts_groups_one_source_to_many_canonicals():
    summary = summarize_source_canonical_url_conflicts(
        [
            {"id": "a", "url": "example.com/a/", "canonical_url": "https://example.com/one"},
            {"id": "b", "url": "https://example.com/a", "canonical_url": "https://example.com/two/"},
            {"id": "c", "url": "https://example.com/a", "canonical_url": "https://example.com/one/"},
        ],
        sample_limit=1,
    )

    assert summary["canonical_conflict_count"] == 0
    assert summary["url_conflict_count"] == 1
    assert summary["conflict_groups"] == [
        {
            "type": "url",
            "url": "https://example.com/a",
            "canonical_urls": ["https://example.com/one", "https://example.com/two"],
            "count": 2,
        }
    ]
    assert len(summary["samples"]) == 1
