from __future__ import annotations

from graph.rag.answer_recommendation_strength import analyze_answer_recommendation_strength


def test_classifies_none_weak_moderate_and_strong_language():
    report = analyze_answer_recommendation_strength("You could retry. Consider batching. You must cite the source.")

    assert report["recommendation_count"] == 3
    assert report["strongest_level"] == "strong"
    assert len(report["unsupported_strong_recommendations"]) == 1


def test_treats_strong_recommendation_with_citation_or_limitation_as_supported():
    report = analyze_answer_recommendation_strength(
        "You should replace the source because the study is outdated [2]. Do not use it unless the scope matches."
    )

    assert report["strongest_level"] == "strong"
    assert report["unsupported_strong_recommendations"] == []


def test_no_recommendations_returns_none():
    report = analyze_answer_recommendation_strength("The answer summarizes three papers.")

    assert report["recommendation_count"] == 0
    assert report["strongest_level"] == "none"
