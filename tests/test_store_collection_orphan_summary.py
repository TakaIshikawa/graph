from graph.store import summarize_collection_orphans


def test_collection_orphans_empty_store():
    assert summarize_collection_orphans([]) == {
        "total_collections": 0,
        "empty_collection_count": 0,
        "empty_collection_ids": [],
        "dangling_reference_count": 0,
        "dangling_references": [],
    }


def test_collection_orphans_reports_empty_and_populated_collections():
    report = summarize_collection_orphans(
        [
            {"id": "empty", "members": []},
            {"id": "full", "member_ids": ["u1"]},
        ]
    )

    assert report["empty_collection_count"] == 1
    assert report["empty_collection_ids"] == ["empty"]


def test_collection_orphans_reports_dangling_unit_collection_references():
    report = summarize_collection_orphans(
        [{"id": "known", "unit_ids": ["u1"]}],
        [{"id": "u1", "collection_ids": ["known", "missing"]}],
    )

    assert report["dangling_reference_count"] == 1
    assert report["dangling_references"] == [{"unit_id": "u1", "collection_id": "missing"}]
