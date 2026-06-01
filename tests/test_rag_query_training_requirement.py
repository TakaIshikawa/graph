from graph.rag.query_training_requirement import detect_query_training_requirements


def test_detects_training_categories():
    rows = detect_query_training_requirements(
        "Need onboarding, a training plan, documentation, runbook, playbook, and user education."
    )

    assert rows == [
        {"matched_text": "documentation", "category": "documentation", "severity": "medium"},
        {"matched_text": "onboarding", "category": "onboarding", "severity": "medium"},
        {"matched_text": "playbook", "category": "playbook", "severity": "medium"},
        {"matched_text": "runbook", "category": "runbook", "severity": "high"},
        {"matched_text": "training plan", "category": "training_plan", "severity": "high"},
        {"matched_text": "user education", "category": "user_education", "severity": "medium"},
    ]


def test_deduplicates_repeated_category_matches():
    assert detect_query_training_requirements("Create DOCS and documentation.") == [
        {"matched_text": "DOCS", "category": "documentation", "severity": "medium"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_training_requirements("Estimate infrastructure cost.") == []
