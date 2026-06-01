from graph.rag.query_human_review_requirement import detect_query_human_review_requirements


def test_detects_human_review_categories():
    rows = detect_query_human_review_requirements(
        "Require human approval, human-in-the-loop, manual signoff, escalation to human, and expert validation."
    )

    assert rows == [
        {"matched_text": "human approval", "category": "approval", "severity": "high"},
        {"matched_text": "escalation to human", "category": "escalation", "severity": "high"},
        {"matched_text": "expert validation", "category": "expert_validation", "severity": "high"},
        {"matched_text": "human-in-the-loop", "category": "human_in_the_loop", "severity": "high"},
        {"matched_text": "manual signoff", "category": "manual_signoff", "severity": "high"},
    ]


def test_matches_mixed_case_review_phrases():
    assert detect_query_human_review_requirements("Need HITL and EXPERT   REVIEW.") == [
        {"matched_text": "EXPERT REVIEW", "category": "expert_validation", "severity": "high"},
        {"matched_text": "HITL", "category": "human_in_the_loop", "severity": "high"},
    ]


def test_casual_people_mentions_do_not_match():
    assert detect_query_human_review_requirements("Explain how people use the workflow.") == []
