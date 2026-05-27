from __future__ import annotations

from graph.rag.result_claim_traceability import analyze_result_claim_traceability


def test_result_claim_traceability_scores_rich_results_higher():
    report = analyze_result_claim_traceability(
        [
            {
                "id": "rich",
                "title": "Trial results",
                "author": "Dr. Lee",
                "published_at": "2024-02-01",
                "url": "https://example.org/trial",
                "snippet": 'The paper reports "events fell by 20 percent" [1].',
            },
            {"id": "weak", "snippet": "This says events fell."},
        ]
    )

    assert report["weak_traceability_count"] == 1
    assert report["results"][0]["traceability_score"] == 6
    assert report["results"][1]["missing_signals"] == ["citation", "url", "quote", "title", "author", "date"]


def test_result_claim_traceability_handles_plain_strings():
    report = analyze_result_claim_traceability(["Unsupported claim without metadata."])

    assert report["total_results"] == 1
    assert report["results"][0]["result_id"] == "result-1"
    assert report["weak_traceability_count"] == 1
