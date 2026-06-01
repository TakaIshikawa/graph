from graph.rag.query_audit_logging_requirement import detect_query_audit_logging_requirements


def test_detects_audit_logging_categories():
    rows = detect_query_audit_logging_requirements(
        "Need audit logs, access logs, traceability, evidence trail, immutable logs, and who-did-what records."
    )

    assert rows == [
        {"matched_text": "access logs", "category": "access_logs", "severity": "high"},
        {"matched_text": "audit logs", "category": "audit_logs", "severity": "high"},
        {"matched_text": "evidence trail", "category": "evidence_trail", "severity": "medium"},
        {"matched_text": "immutable logs", "category": "immutable_logs", "severity": "high"},
        {"matched_text": "traceability", "category": "traceability", "severity": "high"},
        {"matched_text": "who-did-what", "category": "who_did_what", "severity": "high"},
    ]


def test_handles_access_audit_and_immutable_logs():
    assert detect_query_audit_logging_requirements("Show TAMPER   PROOF   LOGS and LOG ACCESS.") == [
        {"matched_text": "LOG ACCESS", "category": "access_logs", "severity": "high"},
        {"matched_text": "TAMPER PROOF LOGS", "category": "immutable_logs", "severity": "high"},
    ]


def test_generic_logging_is_not_enough():
    assert detect_query_audit_logging_requirements("Log debug output while developing.") == []
