from graph.rag.query_governance_approval_requirement import detect_query_governance_approval_requirements


def test_detects_governance_approval_categories():
    rows = detect_query_governance_approval_requirements(
        "Require approval, legal review, security review, change advisory board, procurement, and stakeholder signoff."
    )

    assert rows == [
        {"matched_text": "approval", "category": "approval", "severity": "medium"},
        {"matched_text": "change advisory board", "category": "change_review", "severity": "high"},
        {"matched_text": "legal review", "category": "legal_review", "severity": "high"},
        {"matched_text": "procurement", "category": "procurement", "severity": "medium"},
        {"matched_text": "security review", "category": "security_review", "severity": "high"},
        {"matched_text": "stakeholder signoff", "category": "stakeholder_signoff", "severity": "medium"},
    ]


def test_governance_deduplicates_review_type_categories():
    assert detect_query_governance_approval_requirements("Legal review and counsel review are required.") == [
        {"matched_text": "Legal review", "category": "legal_review", "severity": "high"}
    ]


def test_governance_empty_and_unrelated_queries_return_empty_list():
    assert detect_query_governance_approval_requirements("") == []
    assert detect_query_governance_approval_requirements("Summarize the implementation timeline.") == []
