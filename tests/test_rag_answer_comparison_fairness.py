from __future__ import annotations

from graph.rag.answer_comparison_fairness import audit_answer_comparison_fairness


def test_detects_comparators_and_criteria_cues():
    report = audit_answer_comparison_fairness("Alpha vs Beta based on cost and latency. Alpha is lower cost; Beta is faster.", evidence=[{"content": "Alpha cost. Beta latency."}])

    assert report["comparators"] == ["Alpha", "Beta"]
    assert report["matched_cues"]["criteria"] == ["based on", "cost", "latency"]
    assert report["warnings"] == []


def test_flags_unsupported_winner_and_missing_criteria():
    report = audit_answer_comparison_fairness("Alpha vs Beta. Alpha is best and superior.")

    assert "missing_comparator_criteria" in report["warnings"]
    assert "unsupported_winner_language" in report["warnings"]
    assert report["fairness_score"] < 1.0


def test_flags_uneven_evidence_mentions():
    report = audit_answer_comparison_fairness("Alpha vs Beta using accuracy. Alpha wins. Alpha has accuracy. Alpha has support.")

    assert "one_sided_comparison" in report["warnings"]
