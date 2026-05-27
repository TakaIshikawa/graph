from graph.rag.answer_condition_coverage import audit_answer_condition_coverage


def test_required_condition_full_coverage_returns_empty_list():
    assert audit_answer_condition_coverage("Use it when users opt in.", ["users opt in"]) == []


def test_required_condition_partial_coverage_reports_missing_items():
    findings = audit_answer_condition_coverage("Use it when users opt in.", ["users opt in", "admin approval"])

    assert findings == [
        {
            "missing_condition": "admin approval",
            "severity": "medium",
            "recommendation": "Mention the required condition: admin approval.",
        }
    ]


def test_required_condition_matching_ignores_case_and_punctuation():
    assert audit_answer_condition_coverage("Requires ADMIN approval.", ["admin approval"]) == []


def test_empty_required_conditions_returns_empty_list():
    assert audit_answer_condition_coverage("Answer.", []) == []
