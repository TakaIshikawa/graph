from __future__ import annotations

from graph.rag.evidence_negative_results import detect_evidence_negative_results


def test_classifies_negative_null_failed_inconclusive_and_regression_evidence():
    report = detect_evidence_negative_results(
        [
            {"id": "a", "content": "The study finding was a negative result with no effect."},
            {"id": "b", "content": "The test failed and the metric regressed."},
            {"id": "c", "content": "The analysis was inconclusive."},
        ]
    )

    assert report["category_counts"] == {"failed": 1, "inconclusive": 1, "negative": 1, "null": 1, "regression": 1}
    assert [item["result_id"] for item in report["items"]] == ["a", "b", "c"]


def test_avoids_ordinary_negation_without_result_context():
    report = detect_evidence_negative_results([{"content": "The feature is not enabled for admins."}])

    assert report["items"] == []
    assert all(count == 0 for count in report["category_counts"].values())
