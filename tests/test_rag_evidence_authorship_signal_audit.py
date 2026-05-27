from graph.rag.evidence_authorship_signal import analyze_evidence_authorship_signals


def test_recognizes_top_level_and_metadata_authorship_fields():
    report = analyze_evidence_authorship_signals(
        [{"id": "a", "author": "Ada"}, {"id": "b", "metadata": {"organization": "WHO"}}]
    )

    assert report["authored_count"] == 2
    assert report["by_author"] == {"Ada": 1, "WHO": 1}


def test_missing_authorship_items_use_stable_ids_or_indexes():
    report = analyze_evidence_authorship_signals([{"id": "a"}, {"title": "Untitled"}])

    assert report["missing_authorship_count"] == 2
    assert report["missing_items"] == [{"item_id": "a", "index": 0}, {"item_id": "result-2", "index": 1}]


def test_ratio_is_deterministic():
    report = analyze_evidence_authorship_signals([{"author": "Ada"}, {}])

    assert report["total_items"] == 2
    assert report["authored_ratio"] == 0.5
