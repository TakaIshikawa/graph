from graph.rag.query_observability_requirement import detect_query_observability_requirements


def test_detects_observability_requirement_categories():
    rows = detect_query_observability_requirements(
        "Need logging, metrics, tracing, alerting, dashboards, SLOs, error budgets, and audit trails."
    )

    assert rows == [
        {"matched_text": "alerting", "category": "alerting", "severity": "high"},
        {"matched_text": "audit trails", "category": "audit_trail", "severity": "high"},
        {"matched_text": "dashboards", "category": "dashboard", "severity": "medium"},
        {"matched_text": "error budgets", "category": "error_budget", "severity": "high"},
        {"matched_text": "logging", "category": "logging", "severity": "medium"},
        {"matched_text": "metrics", "category": "metrics", "severity": "medium"},
        {"matched_text": "SLOs", "category": "slo", "severity": "high"},
        {"matched_text": "tracing", "category": "tracing", "severity": "medium"},
    ]


def test_observability_synonyms_and_whitespace_are_normalized():
    rows = detect_query_observability_requirements("Require   telemetry\nand service level objective tracking.")

    assert rows == [
        {"matched_text": "telemetry", "category": "metrics", "severity": "medium"},
        {"matched_text": "service level objective", "category": "slo", "severity": "high"},
    ]


def test_observability_empty_input_returns_empty_list():
    assert detect_query_observability_requirements("") == []
