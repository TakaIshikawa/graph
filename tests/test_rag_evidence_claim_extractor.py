from __future__ import annotations

from graph.rag.evidence_claim_extractor import extract_evidence_claims


def test_extract_evidence_claims_returns_ranked_claim_dictionaries():
    claims = extract_evidence_claims(
        [
            {
                "id": "a",
                "title": "Grid report",
                "url": "https://example.test/report",
                "content": (
                    "Storage costs fell 20% in 2025 because deployments increased. "
                    "This sentence has no strong signal. "
                    "Forecasts improve planning because operators can shift demand."
                ),
            }
        ]
    )

    assert claims[0] == {
        "claim": "Storage costs fell 20% in 2025 because deployments increased.",
        "result_id": "a",
        "source_title": "Grid report",
        "signals": ["number", "date", "causal-cue", "citation"],
        "citation": "https://example.test/report",
    }
    assert claims[1]["signals"] == ["causal-cue", "citation"]


def test_extract_evidence_claims_ranks_cited_numeric_claims_ahead_of_weaker_sentences():
    claims = extract_evidence_claims(
        [
            {"id": "weak", "content": "Better routing improves retrieval quality."},
            {
                "id": "strong",
                "metadata": {"citation_url": "https://example.test"},
                "content": "The benchmark reached 91% accuracy in 2024.",
            },
        ]
    )

    assert [claim["result_id"] for claim in claims] == ["strong", "weak"]


def test_extract_evidence_claims_deduplicates_normalized_sentences_across_results():
    claims = extract_evidence_claims(
        [
            {"id": "a", "content": "Revenue increased 10% in 2024."},
            {"id": "b", "content": "Revenue increased 10 percent in 2024. Revenue increased 10% in 2024!"},
        ]
    )

    assert [claim["result_id"] for claim in claims] == ["a", "b"]
    assert [claim["claim"] for claim in claims] == [
        "Revenue increased 10% in 2024.",
        "Revenue increased 10 percent in 2024.",
    ]


def test_extract_evidence_claims_respects_max_claims():
    claims = extract_evidence_claims(
        [{"id": "a", "content": "One claim has 1 number. Two claim has 2 numbers."}],
        max_claims=1,
    )

    assert len(claims) == 1
