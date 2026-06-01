from graph.rag.query_maintenance_burden_requirement import detect_query_maintenance_burden_requirements


def test_detects_maintenance_burden_categories():
    rows = detect_query_maintenance_burden_requirements(
        "Compare maintenance burden, ongoing upkeep, operational overhead, handoff effort, and long-term support."
    )

    assert rows == [
        {"matched_text": "handoff effort", "category": "handoff_effort", "severity": "medium"},
        {"matched_text": "long-term support", "category": "long_term_support", "severity": "high"},
        {"matched_text": "maintenance burden", "category": "maintenance_burden", "severity": "high"},
        {"matched_text": "ongoing upkeep", "category": "ongoing_upkeep", "severity": "medium"},
        {"matched_text": "operational overhead", "category": "operational_overhead", "severity": "high"},
    ]


def test_normalizes_whitespace_and_deduplicates_categories():
    assert detect_query_maintenance_burden_requirements("Need   OPS   OVERHEAD and operational overhead.") == [
        {"matched_text": "OPS OVERHEAD", "category": "operational_overhead", "severity": "high"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_maintenance_burden_requirements("Rank options by user satisfaction.") == []
