from graph.rag.query_security_control_requirement import detect_query_security_control_requirements


def test_detects_multiple_security_control_categories():
    rows = detect_query_security_control_requirements(
        "Require encryption, MFA, audit logging, least privilege, API keys handling, and access controls."
    )

    assert rows == [
        {"matched_text": "access controls", "category": "access_control", "severity": "high"},
        {"matched_text": "audit logging", "category": "audit_logging", "severity": "high"},
        {"matched_text": "encryption", "category": "encryption", "severity": "high"},
        {"matched_text": "least privilege", "category": "least_privilege", "severity": "high"},
        {"matched_text": "MFA", "category": "mfa", "severity": "high"},
        {"matched_text": "API keys", "category": "secrets_handling", "severity": "high"},
    ]


def test_deduplicates_security_control_categories_with_earliest_match():
    assert detect_query_security_control_requirements("Use encrypted storage and encryption everywhere.") == [
        {"matched_text": "encrypted", "category": "encryption", "severity": "high"}
    ]


def test_security_control_detection_is_case_insensitive():
    assert detect_query_security_control_requirements("Require Multi-Factor Authentication.") == [
        {"matched_text": "Multi-Factor Authentication", "category": "mfa", "severity": "high"}
    ]


def test_security_control_empty_and_unrelated_queries_return_empty_list():
    assert detect_query_security_control_requirements("") == []
    assert detect_query_security_control_requirements("Compare release dates and market share.") == []
