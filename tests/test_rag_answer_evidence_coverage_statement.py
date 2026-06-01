from graph.rag.answer_evidence_coverage_statement import audit_answer_evidence_coverage_statement


def test_multi_source_answer_without_coverage_statement_is_flagged():
    summary = audit_answer_evidence_coverage_statement("The product supports SSO.", evidence=[{"id": "a"}, {"id": "b"}])

    assert summary["evidence_count"] == 2
    assert summary["missing_coverage_statement"] is True


def test_present_coverage_statement_is_detected():
    summary = audit_answer_evidence_coverage_statement(
        "Across sources, the retrieved evidence supports SSO.", evidence=[{"id": "a"}, {"id": "b"}]
    )

    assert summary["has_coverage_statement"] is True
    assert summary["matched_phrases"] == ["across sources", "the retrieved evidence"]
    assert summary["missing_coverage_statement"] is False


def test_single_evidence_record_does_not_require_breadth_statement():
    assert audit_answer_evidence_coverage_statement("The source supports SSO.", evidence=[{"id": "a"}])["missing_coverage_statement"] is False
