from __future__ import annotations

from graph.rag.query_accessibility_compliance_requirement import detect_query_accessibility_compliance_requirement


def test_detect_query_accessibility_compliance_requirement_extracts_terms_and_standards():
    result = detect_query_accessibility_compliance_requirement(
        "Verify WCAG 2.2 and ADA compliance for screen reader support, keyboard navigation, and alt text."
    )

    assert result["requires_accessibility_compliance"] is True
    assert result["accessibility_terms"] == ["wcag", "ada", "screen_reader", "keyboard_navigation", "alt_text"]
    assert result["matched_phrases"] == ["WCAG 2.2", "ADA", "screen reader", "keyboard navigation", "alt text"]
    assert result["standards"] == ["WCAG", "ADA"]
    assert result["confidence"] == 0.95


def test_detect_query_accessibility_compliance_requirement_separates_recommendations():
    result = detect_query_accessibility_compliance_requirement(
        "Need Section 508 evidence for captions, color contrast, reduced motion, and accessible PDF output."
    )

    assert result["standards"] == ["Section 508"]
    assert result["recommendations"] == [
        "retrieve_named_accessibility_standards_documents",
        "retrieve_implementation_evidence_for_accessibility_features",
        "include_accessible_pdf_or_document_evidence",
    ]


def test_detect_query_accessibility_compliance_requirement_returns_non_match():
    assert detect_query_accessibility_compliance_requirement("Compare database indexing options.") == {
        "requires_accessibility_compliance": False,
        "accessibility_terms": [],
        "matched_phrases": [],
        "standards": [],
        "recommendations": [],
        "confidence": 0.0,
    }
