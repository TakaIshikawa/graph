from graph.rag.query_right_to_deletion_requirement import detect_query_right_to_deletion_requirements


def test_detects_right_to_deletion_and_erasure_queries():
    rows = detect_query_right_to_deletion_requirements(
        "Support the right to deletion and right to erasure for all user records."
    )

    assert rows == [
        {"matched_text": "right to deletion", "category": "right_to_deletion", "requirement_strength": "high"},
        {"matched_text": "right to erasure", "category": "right_to_erasure", "requirement_strength": "high"},
    ]


def test_detects_deletion_sla_purge_confirmation_and_account_deletion():
    rows = detect_query_right_to_deletion_requirements(
        "Require account deletion, delete customer data within 30 days, and purge confirmation."
    )

    assert rows == [
        {"matched_text": "account deletion", "category": "account_deletion", "requirement_strength": "high"},
        {"matched_text": "delete customer data", "category": "customer_data_deletion", "requirement_strength": "high"},
        {"matched_text": "delete customer data within", "category": "deletion_sla", "requirement_strength": "high"},
        {"matched_text": "purge confirmation", "category": "purge_confirmation", "requirement_strength": "high"},
    ]


def test_unrelated_retention_or_archive_queries_do_not_match():
    assert detect_query_right_to_deletion_requirements("Compare retention periods and archive storage costs.") == []
