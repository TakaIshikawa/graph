from __future__ import annotations

from graph.rag.query_conflict_of_interest_requirement import detect_query_conflict_of_interest_requirements


def test_conflict_of_interest_requirement_detects_funding_sponsor_and_affiliation():
    report = detect_query_conflict_of_interest_requirements(
        "Check funding sources, industry sponsors, author affiliations, and conflicts of interest."
    )

    assert report["requires_conflict_of_interest_check"] is True
    assert report["categories"] == ["funding", "sponsor", "affiliation", "conflict_of_interest"]


def test_conflict_of_interest_requirement_preserves_matched_phrases():
    report = detect_query_conflict_of_interest_requirements("Prefer evidence from independent sources with disclosures.")

    assert report["categories"] == ["disclosure", "independence"]
    assert report["matched_phrases"] == ["disclosures", "independent"]


def test_conflict_of_interest_requirement_does_not_overclassify_independent():
    report = detect_query_conflict_of_interest_requirements("I need an independent implementation plan.")

    assert report["requires_conflict_of_interest_check"] is False
    assert report["requirements"] == []
