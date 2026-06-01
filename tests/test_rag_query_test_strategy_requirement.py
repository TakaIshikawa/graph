from graph.rag.query_test_strategy_requirement import detect_query_test_strategy_requirements


def test_detects_test_strategy_categories():
    rows = detect_query_test_strategy_requirements(
        "Include a test plan, acceptance tests, regression tests, validation criteria, QA, and verification steps."
    )

    assert rows == [
        {"matched_text": "acceptance tests", "category": "acceptance_tests", "severity": "high"},
        {"matched_text": "QA", "category": "qa", "severity": "medium"},
        {"matched_text": "regression tests", "category": "regression_tests", "severity": "high"},
        {"matched_text": "test plan", "category": "test_plan", "severity": "high"},
        {"matched_text": "validation criteria", "category": "validation_criteria", "severity": "high"},
        {"matched_text": "verification steps", "category": "verification_steps", "severity": "medium"},
    ]


def test_deduplicates_repeated_category_matches():
    assert detect_query_test_strategy_requirements("Need TESTING   STRATEGY and a test plan.") == [
        {"matched_text": "TESTING STRATEGY", "category": "test_plan", "severity": "high"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_test_strategy_requirements("Summarize customer feedback.") == []
