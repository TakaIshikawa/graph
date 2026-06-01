from graph.rag.query_latency_budget_requirement import detect_query_latency_budget_requirements


def test_detects_latency_budget_categories():
    rows = detect_query_latency_budget_requirements(
        "Set latency, response time, timeout, p95, real-time, and performance budget constraints."
    )

    assert rows == [
        {"matched_text": "latency", "category": "latency", "severity": "high"},
        {"matched_text": "p95", "category": "percentile_latency", "severity": "high"},
        {"matched_text": "performance budget", "category": "performance_budget", "severity": "medium"},
        {"matched_text": "real-time", "category": "realtime", "severity": "high"},
        {"matched_text": "response time", "category": "response_time", "severity": "high"},
        {"matched_text": "timeout", "category": "timeout", "severity": "high"},
    ]


def test_handles_mixed_case_and_multiple_signals():
    assert detect_query_latency_budget_requirements("Need P99 and RESPONSE   TIME.") == [
        {"matched_text": "P99", "category": "percentile_latency", "severity": "high"},
        {"matched_text": "RESPONSE TIME", "category": "response_time", "severity": "high"},
    ]


def test_generic_performance_is_not_enough():
    assert detect_query_latency_budget_requirements("Improve performance and readability.") == []
