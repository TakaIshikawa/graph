from graph.rag.query_sandbox_access_requirement import detect_query_sandbox_access_requirement


def test_detects_sandbox_and_trial_tenant_access_requests():
    result = detect_query_sandbox_access_requirement("Request sandbox access and a trial tenant for evaluation.")

    assert result["requires_sandbox_access"] is True
    assert result["sandbox_terms"] == ["sandbox", "trial_workspace"]
    assert result["confidence"] == "medium"


def test_environment_terms_are_ordered_predictably():
    result = detect_query_sandbox_access_requirement(
        "Provision a staging account with non-production access, isolated test data, and a mock integration."
    )

    assert result["requires_sandbox_access"] is True
    assert result["environment_terms"] == ["isolated_test_data", "mock_integration", "non_production", "staging_account"]
    assert result["confidence"] == "high"


def test_ambiguous_sandbox_mention_without_access_or_evaluation_intent_is_negative():
    assert detect_query_sandbox_access_requirement("Explain the sandbox concept in operating systems.") == {
        "requires_sandbox_access": False,
        "sandbox_terms": [],
        "environment_terms": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }
