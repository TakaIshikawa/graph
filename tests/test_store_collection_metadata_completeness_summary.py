from __future__ import annotations

from graph.store import summarize_collection_metadata_completeness


def test_collection_metadata_completeness_counts_required_key_coverage():
    summary = summarize_collection_metadata_completeness(
        [
            {"id": "c1", "title": "One", "metadata": {"description": "Desc", "source": "rss", "updated_at": "2024-01-01"}},
            {"id": "c2", "metadata": {"title": "", "source": "rss"}},
            {"id": "c3", "metadata": {}},
        ]
    )

    assert summary["total_collections"] == 3
    assert summary["required_key_coverage"]["source"]["present_count"] == 2
    assert summary["overall_coverage_ratio"] == "0.42"
    assert summary["missing_by_collection"][0] == {
        "collection_id": "c2",
        "missing_keys": ["title", "description", "updated_at"],
    }


def test_collection_metadata_completeness_allows_required_keys():
    summary = summarize_collection_metadata_completeness([{"id": "c1", "metadata": {"owner": "ops"}}], required_keys=("owner",))

    assert summary["overall_coverage_ratio"] == "1.00"
