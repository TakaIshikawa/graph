from __future__ import annotations

from graph.rag.evidence_primary_quote_selector import select_primary_evidence_quotes


def test_evidence_primary_quote_selector_ranks_by_query_overlap():
    payload = select_primary_evidence_quotes(
        "api latency budget",
        [
            {"id": "weak", "text": "The release shipped successfully."},
            {"id": "strong", "text": "The API latency budget improved after caching."},
        ],
        limit=1,
    )

    assert payload["selected_quotes"][0]["result_id"] == "strong"
    assert "query term overlap" in payload["selected_quotes"][0]["reasons"]


def test_evidence_primary_quote_selector_uses_citation_availability():
    payload = select_primary_evidence_quotes(
        "release",
        [{"text": "The release shipped."}, {"url": "https://example.com", "text": "The release shipped."}],
        limit=1,
    )

    assert payload["selected_quotes"][0]["result_id"] == "result-2"
    assert "citation available" in payload["selected_quotes"][0]["reasons"]


def test_evidence_primary_quote_selector_stable_tie_breaking_and_length():
    payload = select_primary_evidence_quotes(
        "release",
        [{"id": "a", "text": "Release " + "x" * 50}, {"id": "b", "text": "Release " + "y" * 50}],
        limit=2,
        max_quote_length=20,
    )

    assert [row["result_id"] for row in payload["selected_quotes"]] == ["a", "b"]
    assert len(payload["selected_quotes"][0]["quote"]) <= 20
