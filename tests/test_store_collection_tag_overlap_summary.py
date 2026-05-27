from __future__ import annotations

from graph.store import summarize_collection_tag_overlap


def test_summarize_collection_tag_overlap_returns_pairwise_jaccard():
    summary = summarize_collection_tag_overlap([
        {"id": "a", "name": "A", "tags": ["one", "two"]},
        {"id": "b", "name": "B", "tags": ["one", "three"]},
        {"id": "c", "name": "C", "tags": ["none"]},
    ])

    assert summary["overlaps"] == [
        {"collection_id_a": "a", "collection_name_a": "A", "collection_id_b": "b", "collection_name_b": "B", "shared_tag_count": 1, "jaccard": 0.3333, "shared_tag_samples": ["one"]}
    ]
