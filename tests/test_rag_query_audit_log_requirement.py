from graph.rag.query_audit_log_requirement import detect_query_audit_log_requirement


def test_detects_audit_log_requirement_categories_sorted_by_category():
    result = detect_query_audit_log_requirement(
        "Need audit trail evidence, event logs, immutable logs, log retention, "
        "admin activity logs, exportability, and audit logs."
    )

    assert result["has_audit_log_requirement"] is True
    assert result["requirements"] == [
        {"category": "admin_activity", "matched_text": "admin activity logs", "severity": "high"},
        {"category": "audit_log", "matched_text": "audit logs", "severity": "high"},
        {"category": "audit_trail", "matched_text": "audit trail", "severity": "high"},
        {"category": "event_log", "matched_text": "event logs", "severity": "medium"},
        {"category": "exportability", "matched_text": "exportability", "severity": "medium"},
        {"category": "immutability", "matched_text": "immutable logs", "severity": "high"},
        {"category": "retention", "matched_text": "log retention", "severity": "high"},
    ]


def test_detects_wording_variants_and_normalizes_spacing():
    result = detect_query_audit_log_requirement(
        "Require tamper proof audit logs, retain audit logs for 7 years, "
        "EVENT   HISTORY, privileged user activity logs, and download audit logs."
    )

    assert [(row["category"], row["matched_text"]) for row in result["requirements"]] == [
        ("admin_activity", "privileged user activity logs"),
        ("audit_log", "audit logs"),
        ("event_log", "EVENT HISTORY"),
        ("exportability", "download audit logs"),
        ("immutability", "tamper proof audit logs"),
        ("retention", "retain audit logs for"),
    ]


def test_unrelated_application_logging_mentions_do_not_trigger():
    assert detect_query_audit_log_requirement("Log debug output and application errors for troubleshooting.") == {
        "has_audit_log_requirement": False,
        "requirements": [],
    }
