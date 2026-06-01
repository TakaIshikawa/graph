from graph.rag.query_audience_expertise_requirement import detect_query_audience_expertise_requirements


def test_detects_audience_expertise_categories():
    rows = detect_query_audience_expertise_requirements(
        "Explain for beginners, executives, developers, experts, non-technical readers, and specialists."
    )

    assert rows == [
        {"matched_text": "beginners", "category": "beginner", "severity": "medium"},
        {"matched_text": "developers", "category": "developer", "severity": "medium"},
        {"matched_text": "executives", "category": "executive", "severity": "medium"},
        {"matched_text": "experts", "category": "expert", "severity": "high"},
        {"matched_text": "non-technical", "category": "non_technical", "severity": "high"},
        {"matched_text": "specialists", "category": "specialist", "severity": "high"},
    ]


def test_is_case_insensitive_and_normalizes_whitespace():
    assert detect_query_audience_expertise_requirements("Use   PLAIN   ENGLISH for a DATA   SCIENTIST.") == [
        {"matched_text": "PLAIN ENGLISH", "category": "non_technical", "severity": "high"},
        {"matched_text": "DATA SCIENTIST", "category": "specialist", "severity": "high"},
    ]


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_audience_expertise_requirements("Summarize the release notes with citations.") == []
