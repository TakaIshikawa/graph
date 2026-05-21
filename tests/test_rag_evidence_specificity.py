from __future__ import annotations

from graph.rag.evidence_specificity import score_evidence_specificity


def test_evidence_specificity_scores_rich_evidence_above_generic_summary():
    report = score_evidence_specificity(
        [
            {"id": "generic", "snippet": "Project Alpha improved outcomes and users liked the result."},
            {
                "id": "rich",
                "snippet": 'According to Acme Research on 2026-04-12, "conversion rose by 18%" using a 240-user survey. See https://example.test/report.',
            },
        ]
    )

    rows = {row["result_id"]: row for row in report["results"]}
    assert rows["rich"]["score"] > rows["generic"]["score"]
    assert rows["rich"]["bucket"] == "high"
    assert set(rows["rich"]["signals"]) == {"number", "date", "url", "quoted_text", "named_entity", "method_detail"}
    assert rows["generic"]["bucket"] == "low"


def test_evidence_specificity_aggregate_buckets_are_stable():
    assert score_evidence_specificity([])["aggregate_bucket"] == "empty"
    assert score_evidence_specificity([{"id": "low", "text": "Useful general context."}])["aggregate_bucket"] == "empty"
    assert score_evidence_specificity([{"id": "medium", "text": "Acme Research measured 42 cases."}])["aggregate_bucket"] == "medium"
    assert score_evidence_specificity(
        [{"id": "high", "text": 'Acme Research measured 42 cases on 2026-01-10 using table 2: "39 passed." https://example.test'}]
    )["aggregate_bucket"] == "high"


def test_evidence_specificity_reports_counts_and_warnings():
    report = score_evidence_specificity([{"id": "a", "snippet": "General background only."}])

    assert report["bucket_counts"] == {"empty": 1, "low": 0, "medium": 0, "high": 0}
    assert "low_specificity_result_set" in report["warnings"]
    assert report["results"][0]["result_id"] == "a"
    assert report["results"][0]["signals"] == []
