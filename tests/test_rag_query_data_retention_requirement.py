from graph.rag.query_data_retention_requirement import detect_query_data_retention_requirements


def test_detects_data_retention_requirement_categories():
    rows = detect_query_data_retention_requirements(
        "Need retention period, deletion window, archival rules, purge policy, and recordkeeping obligations."
    )

    assert rows == [
        {"matched_text": "archival", "category": "archive"},
        {"matched_text": "deletion window", "category": "deletion"},
        {"matched_text": "purge policy", "category": "purge"},
        {"matched_text": "recordkeeping", "category": "recordkeeping"},
        {"matched_text": "retention period", "category": "retention"},
    ]


def test_data_retention_deduplicates_and_preserves_stable_ordering():
    assert detect_query_data_retention_requirements("Retain for 7 years and data retention controls.") == [
        {"matched_text": "Retain for", "category": "retention"}
    ]


def test_data_retention_non_requirement_wording_returns_empty_list():
    assert detect_query_data_retention_requirements("Summarize storage vendors and query speed.") == []
