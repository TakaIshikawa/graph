from graph.rag.query_contract_termination_requirement import detect_query_contract_termination_requirements


def test_detects_contract_termination_categories_sorted():
    rows = detect_query_contract_termination_requirements(
        "Review vendor contract termination rights, termination for convenience, termination for cause, "
        "30 days notice, post-termination data return, survival clauses, transition assistance, and exit fees."
    )

    assert rows == [
        {"matched_text": "post-termination data return", "category": "data_return_deletion", "requirement_strength": "high"},
        {"matched_text": "exit fees", "category": "exit_fees", "requirement_strength": "medium"},
        {"matched_text": "30 days notice", "category": "notice_period", "requirement_strength": "high"},
        {"matched_text": "survival clauses", "category": "survival_clauses", "requirement_strength": "medium"},
        {"matched_text": "termination for cause", "category": "termination_for_cause", "requirement_strength": "high"},
        {
            "matched_text": "termination for convenience",
            "category": "termination_for_convenience",
            "requirement_strength": "high",
        },
        {"matched_text": "termination rights", "category": "termination_rights", "requirement_strength": "high"},
        {"matched_text": "transition assistance", "category": "transition_assistance", "requirement_strength": "medium"},
    ]


def test_requires_contract_vendor_or_procurement_context():
    assert detect_query_contract_termination_requirements("Need termination for cause and 30 days notice.") == []


def test_negative_non_contract_lifecycle_wording_returns_empty_results():
    assert detect_query_contract_termination_requirements("Plan app lifecycle decommissioning and data deletion.") == []
