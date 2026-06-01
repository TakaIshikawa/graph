from graph.rag.query_reliability_requirement import detect_query_reliability_requirements


def test_detects_reliability_requirement_categories():
    rows = detect_query_reliability_requirements(
        "Need uptime, availability, failover, redundant storage, retries, disaster recovery, RTO, and RPO."
    )

    assert rows == [
        {"matched_text": "availability", "category": "availability", "severity": "high"},
        {"matched_text": "disaster recovery", "category": "disaster_recovery", "severity": "high"},
        {"matched_text": "failover", "category": "failover", "severity": "high"},
        {"matched_text": "redundant", "category": "redundancy", "severity": "medium"},
        {"matched_text": "retries", "category": "retry", "severity": "medium"},
        {"matched_text": "RPO", "category": "rpo", "severity": "high"},
        {"matched_text": "RTO", "category": "rto", "severity": "high"},
        {"matched_text": "uptime", "category": "uptime", "severity": "high"},
    ]


def test_reliability_acronym_matches_and_deduplicates_by_category():
    assert detect_query_reliability_requirements("RTO and recovery time objective under one hour.") == [
        {"matched_text": "RTO", "category": "rto", "severity": "high"}
    ]


def test_reliability_unrelated_operational_text_returns_empty_list():
    assert detect_query_reliability_requirements("List deployment owners and meeting cadence.") == []
