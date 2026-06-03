from graph.rag.query_data_subject_access_request_requirement import detect_query_data_subject_access_request_requirements


def test_detects_dsar_subject_access_and_request_portal_requirements():
    rows = detect_query_data_subject_access_request_requirements(
        "Need DSAR intake, subject access request routing, and a privacy request portal."
    )

    assert rows == [
        {"matched_text": "DSAR", "category": "dsar", "requirement_strength": "high"},
        {"matched_text": "privacy request portal", "category": "request_portal", "requirement_strength": "medium"},
        {"matched_text": "subject access request", "category": "subject_access_request", "requirement_strength": "high"},
    ]


def test_detects_workflow_verification_deadline_and_fulfillment_evidence():
    rows = detect_query_data_subject_access_request_requirements(
        "For GDPR access requests, document the access request workflow, identity verification for requests, "
        "respond to access requests within 30 days, and fulfillment evidence."
    )

    assert rows == [
        {"matched_text": "access request workflow", "category": "access_request_workflow", "requirement_strength": "medium"},
        {"matched_text": "fulfillment evidence", "category": "fulfillment_evidence", "requirement_strength": "medium"},
        {"matched_text": "identity verification for requests", "category": "identity_verification", "requirement_strength": "high"},
        {"matched_text": "respond to access requests within", "category": "response_deadline", "requirement_strength": "high"},
    ]


def test_generic_api_or_account_access_requests_do_not_match():
    assert detect_query_data_subject_access_request_requirements("Add API access requests for partner accounts.") == []
    assert detect_query_data_subject_access_request_requirements("Review account access request approvals for admins.") == []
