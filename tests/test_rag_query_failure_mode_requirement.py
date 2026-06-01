from graph.rag.query_failure_mode_requirement import detect_query_failure_mode_requirements


def test_detects_failure_mode_requirement_categories():
    rows = detect_query_failure_mode_requirements(
        "List risks, failure modes, pitfalls, edge cases, and a rollback plan."
    )

    assert rows == [
        {"matched_text": "edge cases", "category": "edge_case", "severity": "medium"},
        {"matched_text": "failure modes", "category": "failure_mode", "severity": "high"},
        {"matched_text": "pitfalls", "category": "pitfall", "severity": "medium"},
        {"matched_text": "risks", "category": "risk", "severity": "high"},
        {"matched_text": "rollback plan", "category": "rollback", "severity": "high"},
    ]


def test_deduplicates_repeated_category_matches():
    assert detect_query_failure_mode_requirements("What risks and what could go wrong?") == [
        {"matched_text": "risks", "category": "risk", "severity": "high"}
    ]
