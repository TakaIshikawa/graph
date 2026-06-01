from graph.rag.query_deprecation_requirement import detect_query_deprecation_requirements


def test_detects_deprecation_categories():
    rows = detect_query_deprecation_requirements(
        "Plan deprecation, sunset, end-of-life, removal timeline, and replacement path."
    )

    assert rows == [
        {"matched_text": "deprecation", "category": "deprecation", "severity": "high"},
        {"matched_text": "end-of-life", "category": "end_of_life", "severity": "high"},
        {"matched_text": "removal timeline", "category": "removal_timeline", "severity": "high"},
        {"matched_text": "replacement path", "category": "replacement_path", "severity": "medium"},
        {"matched_text": "sunset", "category": "sunset", "severity": "high"},
    ]


def test_handles_eol_abbreviation():
    assert detect_query_deprecation_requirements("What is the EOL plan?") == [
        {"matched_text": "EOL", "category": "end_of_life", "severity": "high"}
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_deprecation_requirements("Remove duplicate rows in the report.") == []
