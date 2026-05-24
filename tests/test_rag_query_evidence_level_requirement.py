from __future__ import annotations

from graph.rag.query_evidence_level_requirement import detect_query_evidence_level_requirement


def test_detects_systematic_reviews_and_randomized_trials():
    result = detect_query_evidence_level_requirement("Use systematic reviews and randomized controlled trials.")

    assert result["requires_evidence_level"] is True
    assert result["requirement_labels"] == ["systematic_review", "randomized_trial"]
    assert result["requirements"][0]["evidence_tier"] == "highest"


def test_detects_primary_official_and_peer_reviewed_sources():
    result = detect_query_evidence_level_requirement("Find primary sources, official documentation, and peer-reviewed studies.")

    assert result["requirement_labels"] == ["primary_source", "official_source", "peer_reviewed"]
    assert [row["evidence_category"] for row in result["requirements"]] == ["primary", "official", "scholarly"]


def test_detects_expert_and_anecdotal_evidence():
    result = detect_query_evidence_level_requirement("Include expert opinion and anecdotal evidence.")

    assert result["requirement_labels"] == ["expert_opinion", "anecdotal_evidence"]
    assert result["matched_phrases"] == ["expert opinion", "anecdotal evidence"]


def test_neutral_query_returns_no_requirements():
    result = detect_query_evidence_level_requirement("What is retrieval augmented generation?")

    assert result["requires_evidence_level"] is False
    assert result["requirements"] == []
