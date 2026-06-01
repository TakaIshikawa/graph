from graph.rag.query_persona_requirement import detect_query_persona_requirements


def test_detects_common_persona_requirements_with_stable_labels():
    rows = detect_query_persona_requirements(
        "Summarize this for executives, explain caveats for clinicians, and add notes for policy makers."
    )

    assert rows == [
        {"persona": "clinician", "matched_text": "for clinicians", "severity": "high"},
        {"persona": "executive", "matched_text": "for executives", "severity": "high"},
        {"persona": "policy maker", "matched_text": "for policy makers", "severity": "high"},
    ]


def test_detects_beginner_developer_and_child_wording():
    rows = detect_query_persona_requirements("Make it developer-focused, in plain English, for kids.")

    assert rows == [
        {"persona": "beginner", "matched_text": "in plain English", "severity": "medium"},
        {"persona": "child", "matched_text": "for kids", "severity": "medium"},
        {"persona": "developer", "matched_text": "developer-focused", "severity": "medium"},
    ]


def test_avoids_false_positives_for_unrelated_senior_and_junior_words():
    assert detect_query_persona_requirements("Compare senior debt with junior debt in the capital stack.") == []
