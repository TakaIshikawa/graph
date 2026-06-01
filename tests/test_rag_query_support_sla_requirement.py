from graph.rag.query_support_sla_requirement import detect_query_support_sla_requirements


def test_detects_support_sla_categories():
    rows = detect_query_support_sla_requirements(
        "Specify SLA, support window, response time, escalation, incident support, and uptime commitment."
    )

    assert rows == [
        {"matched_text": "escalation", "category": "escalation", "severity": "high"},
        {"matched_text": "incident support", "category": "incident_support", "severity": "high"},
        {"matched_text": "response time", "category": "response_time", "severity": "high"},
        {"matched_text": "SLA", "category": "sla", "severity": "high"},
        {"matched_text": "support window", "category": "support_window", "severity": "medium"},
        {"matched_text": "uptime commitment", "category": "uptime_commitment", "severity": "high"},
    ]


def test_matches_case_insensitively():
    assert detect_query_support_sla_requirements("Need SERVICE LEVEL AGREEMENT.") == [
        {"matched_text": "SERVICE LEVEL AGREEMENT", "category": "sla", "severity": "high"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_support_sla_requirements("Support the argument with citations.") == []
