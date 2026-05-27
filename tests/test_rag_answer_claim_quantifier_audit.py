from graph.rag.answer_claim_quantifier import audit_answer_claim_quantifiers


def test_returns_quantifier_findings_with_sentence():
    findings = audit_answer_claim_quantifiers("All users benefit. Many teams still wait.")

    assert findings[0] == {
        "phrase": "All",
        "normalized_quantifier": "all",
        "sentence": "All users benefit.",
        "severity": "high",
        "recommendation": "Qualify broad quantifiers or cite evidence that supports the scope.",
    }
    assert findings[1]["sentence"] == "Many teams still wait."


def test_matching_is_case_insensitive():
    findings = audit_answer_claim_quantifiers("MOST cases improve, but never all.")

    assert [item["normalized_quantifier"] for item in findings] == ["most", "never", "all"]


def test_no_quantifiers_returns_empty_list():
    assert audit_answer_claim_quantifiers("Some teams may benefit.") == []
