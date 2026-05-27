from types import SimpleNamespace

from graph.rag.context_duplicate_source import audit_context_duplicate_sources


def test_duplicate_sources_group_by_source_id_url_then_title():
    report = audit_context_duplicate_sources(
        [
            {"id": "a", "source_id": "s1", "url": "https://x.test/a", "title": "Alpha"},
            {"id": "b", "source_id": "s1", "url": "https://x.test/b", "title": "Beta"},
            {"id": "c", "url": "https://www.example.com/Page/", "title": "Gamma"},
            {"id": "d", "url": "https://example.com/Page", "title": "Delta"},
            {"id": "e", "title": "Same"},
            {"id": "f", "title": "same"},
            {"id": "g", "title": "Unique"},
        ]
    )

    assert report["context_count"] == 7
    assert report["duplicate_group_count"] == 3
    assert report["duplicate_item_count"] == 6
    assert report["diversity_ratio"] == 0.1429
    assert [group["key_type"] for group in report["groups"]] == ["source_id", "title", "url"]


def test_duplicate_sources_support_objects():
    report = audit_context_duplicate_sources([SimpleNamespace(id="a", source_id="s"), SimpleNamespace(id="b", source_id="s")])

    assert report["groups"][0]["item_ids"] == ["a", "b"]
