from graph.rag.evidence_method_signal import classify_evidence_method_signals


def test_counts_multiple_methods():
    report = classify_evidence_method_signals(
        [{"id": "a", "title": "Survey and interview results"}, {"id": "b", "content": "Randomized experiment"}]
    )

    assert report["method_counts"]["survey"] == 1
    assert report["method_counts"]["interview"] == 1
    assert report["method_counts"]["randomized"] == 1
    assert report["method_counts"]["experiment"] == 1


def test_detects_metadata_only_cues():
    report = classify_evidence_method_signals([{"metadata": {"study_type": "case study benchmark"}}])

    assert [item["method"] for item in report["matched_items"]] == ["case_study", "benchmark"]


def test_items_without_cues_do_not_match():
    report = classify_evidence_method_signals([{"title": "Opinion note"}])

    assert report["matched_items"] == []
