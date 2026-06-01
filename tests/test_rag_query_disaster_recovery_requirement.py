from graph.rag.query_disaster_recovery_requirement import detect_query_disaster_recovery_requirements


def test_detects_disaster_recovery_categories():
    rows = detect_query_disaster_recovery_requirements(
        "Need disaster recovery, backup, failover, RTO, RPO, business continuity, and recovery plan."
    )

    assert rows == [
        {"matched_text": "backup", "category": "backup_restore", "severity": "high"},
        {"matched_text": "business continuity", "category": "business_continuity", "severity": "high"},
        {"matched_text": "disaster recovery", "category": "disaster_recovery", "severity": "high"},
        {"matched_text": "failover", "category": "failover", "severity": "high"},
        {"matched_text": "recovery plan", "category": "recovery_plan", "severity": "high"},
        {"matched_text": "RPO", "category": "rpo", "severity": "high"},
        {"matched_text": "RTO", "category": "rto", "severity": "high"},
    ]


def test_detects_failover_and_continuity_planning():
    assert detect_query_disaster_recovery_requirements("Cover FAIL   OVER and continuity planning.") == [
        {"matched_text": "continuity planning", "category": "business_continuity", "severity": "high"},
        {"matched_text": "FAIL OVER", "category": "failover", "severity": "high"},
    ]


def test_unrelated_recovery_wording_does_not_match():
    assert detect_query_disaster_recovery_requirements("Recover deleted text from the prompt.") == []
