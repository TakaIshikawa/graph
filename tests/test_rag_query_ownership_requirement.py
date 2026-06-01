from graph.rag.query_ownership_requirement import detect_query_ownership_requirements


def test_detects_ownership_categories_sorted_by_category():
    rows = detect_query_ownership_requirements(
        "Name the owner, accountable lead, RACI, responsibility matrix, decision maker, and escalation path."
    )

    assert rows == [
        {"matched_text": "accountable", "category": "accountability", "severity": "high"},
        {"matched_text": "decision maker", "category": "decision_maker", "severity": "medium"},
        {"matched_text": "escalation path", "category": "escalation_path", "severity": "high"},
        {"matched_text": "owner", "category": "owner", "severity": "high"},
        {"matched_text": "RACI", "category": "raci", "severity": "medium"},
        {"matched_text": "responsibility matrix", "category": "responsibility_matrix", "severity": "medium"},
    ]


def test_deduplicates_repeated_category_matches():
    assert detect_query_ownership_requirements("Who   DECIDES and decision-maker?") == [
        {"matched_text": "Who DECIDES", "category": "decision_maker", "severity": "medium"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_ownership_requirements("Compare delivery dates.") == []
