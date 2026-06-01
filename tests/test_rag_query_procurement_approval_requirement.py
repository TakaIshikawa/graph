from graph.rag.query_procurement_approval_requirement import detect_query_procurement_approval_requirement


def test_procurement_approval_and_vendor_review_are_detected():
    report = detect_query_procurement_approval_requirement(
        "Require procurement approval, vendor review, RFP, purchase order, and approved vendor list."
    )

    assert report["requires_procurement_approval"] is True
    assert report["approval_terms"] == ["procurement", "vendor_review", "rfp", "purchase_order", "approved_vendor_list"]
    assert report["confidence"] == "medium"


def test_stakeholder_terms_are_deduplicated_and_ordered():
    report = detect_query_procurement_approval_requirement(
        "Procurement process needs legal review, finance approval, security review, and finance review."
    )

    assert report["stakeholder_terms"] == ["legal", "finance", "security"]
    assert report["confidence"] == "high"


def test_non_enterprise_purchasing_language_does_not_trigger_high_confidence():
    assert detect_query_procurement_approval_requirement("Compare purchase options for a personal laptop.") == {
        "requires_procurement_approval": False,
        "approval_terms": [],
        "matched_phrases": [],
        "stakeholder_terms": [],
        "recommendations": [],
        "confidence": "none",
    }
