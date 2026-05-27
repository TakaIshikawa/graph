from __future__ import annotations

from graph.rag.query_evidence_type_requirement import detect_query_evidence_type_requirements


def test_query_evidence_type_requirement_detects_multiple_types():
    report = detect_query_evidence_type_requirements("Use peer-reviewed studies, benchmarks, and official docs.")

    assert report["has_evidence_type_requirement"] is True
    assert report["requirement_labels"] == ["peer_reviewed_study", "benchmark", "official_docs"]
    assert report["requirements"][0]["matched_spans"][0]["text"] == "peer-reviewed"


def test_query_evidence_type_requirement_preserves_matched_spans():
    report = detect_query_evidence_type_requirements("Include case studies and statistics.")

    assert report["requirement_labels"] == ["statistics", "case_study"]
    assert report["requirements"][0]["matched_spans"] == [{"text": "statistics", "start": 25, "end": 35}]


def test_query_evidence_type_requirement_does_not_treat_generic_source_as_specific():
    report = detect_query_evidence_type_requirements("Give me sources for this claim.")

    assert report["has_evidence_type_requirement"] is False
    assert report["requirements"] == []
