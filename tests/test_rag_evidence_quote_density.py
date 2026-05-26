from __future__ import annotations

from graph.rag import score_evidence_quote_density


def test_quote_density_counts_straight_and_curly_quotes():
    result = score_evidence_quote_density(['Alpha "quoted text"', {"snippet": "Beta “curly” end"}])

    assert result == {"total_chars": 35, "quoted_chars": 16, "quote_density": 0.4571, "quote_count": 2, "density_bucket": "medium"}


def test_quote_density_handles_empty_evidence():
    assert score_evidence_quote_density([]) == {"total_chars": 0, "quoted_chars": 0, "quote_density": 0.0, "quote_count": 0, "density_bucket": "none"}


def test_quote_density_buckets_are_deterministic_at_boundaries():
    assert score_evidence_quote_density(['"12"345678'])["density_bucket"] == "low"
    assert score_evidence_quote_density(['"123456"7890'])["density_bucket"] == "high"
