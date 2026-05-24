from __future__ import annotations

from graph.rag.context_contradiction_clusters import cluster_context_contradictions


class Item:
    def __init__(self, id: str, text: str) -> None:
        self.id = id
        self.text = text


def test_clusters_dict_and_object_items_by_shared_terms_and_opposite_cues():
    report = cluster_context_contradictions(
        [
            {"id": "a", "text": "Widget rollout support increased for enterprise customers."},
            Item("b", "Widget rollout support decreased for enterprise customers."),
            {"id": "c", "text": "Unrelated note."},
        ]
    )

    assert report["clusters"][0]["item_ids"] == ["a", "b"]
    assert "widget" in report["clusters"][0]["terms"]
    assert report["clusters"][0]["triggers"] == ["increase/decrease"]
    assert report["unclustered_count"] == 1


def test_no_clusters_without_opposite_cues():
    report = cluster_context_contradictions([{"text": "Policy is available."}, {"text": "Policy is supported."}])

    assert report["clusters"] == []
    assert report["unclustered_count"] == 2
