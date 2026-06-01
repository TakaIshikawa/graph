from __future__ import annotations

from graph.store.relation_evidence_status_code_summary import summarize_relation_evidence_status_codes


def test_summarizes_relation_metadata_and_evidence_status_codes():
    summary = summarize_relation_evidence_status_codes(
        [
            {"id": "r1", "metadata": {"status_code": "200"}, "evidence": [{"status": 301}]},
            {"id": "r2", "evidence": [{"http_status": "404"}, {"response_status": 503}]},
            {"source": "a", "target": "b", "type": "cites", "metadata": {"evidence": {"status_code": "nope"}}},
        ]
    )

    assert summary["total_relations"] == 3
    assert summary["relations_with_status_codes"] == 2
    assert summary["status_counts"] == {"200": 1, "301": 1, "404": 1, "503": 1}
    assert summary["status_class_counts"] == {"2xx": 1, "3xx": 1, "4xx": 1, "5xx": 1}
    assert summary["invalid_count"] == 1
    assert summary["samples"][0] == {"relation_id": "r1", "status_code": 200, "valid": True}


def test_invalid_status_values_are_sampled_deterministically():
    summary = summarize_relation_evidence_status_codes(
        [
            {"source": "s", "target": "t", "type": "rel", "status_code": "999"},
            {"id": "r2", "evidence": [{"status": "bad"}]},
        ],
        sample_limit=2,
    )

    assert summary["invalid_count"] == 2
    assert summary["samples"] == [
        {"relation_id": "s|t|rel", "status_code": "999", "valid": False},
        {"relation_id": "r2", "status_code": "bad", "valid": False},
    ]
