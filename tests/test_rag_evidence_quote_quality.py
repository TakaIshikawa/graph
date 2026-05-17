from __future__ import annotations

from graph.rag import score_evidence_quote_quality


def test_evidence_quote_quality_scores_useful_quotes_higher():
    rows = score_evidence_quote_quality(
        [
            {
                "id": "good",
                "quote": "Battery storage policies improved retention across measured deployments.",
                "url": "https://example.test",
            },
            {"id": "weak", "quote": "maybe"},
        ],
        query="battery retention policy",
    )

    assert rows[0]["evidence_id"] == "good"
    assert rows[0]["quality_score"] > rows[1]["quality_score"]
    assert rows[0]["quality_score"] <= 1.0
    assert "citation_present" in rows[0]["strengths"]
    assert "missing_citation" in rows[1]["warnings"]


def test_evidence_quote_quality_is_bounded_for_plain_strings():
    rows = score_evidence_quote_quality(["Short fragment"], query="battery")

    assert rows[0]["evidence_id"] == "evidence-1"
    assert 0.0 <= rows[0]["quality_score"] <= 1.0
