from graph.rag.query_decommissioning_requirement import detect_query_decommissioning_requirement


def test_detects_system_sunset_with_lifecycle_terms():
    result = detect_query_decommissioning_requirement("Plan to sunset the service, migrate users, archive before removal, and replacement rollout.")

    assert result["requires_decommissioning_plan"] is True
    assert result["decommissioning_terms"] == ["sunset"]
    assert result["lifecycle_terms"] == ["archive", "migrate", "replace"]
    assert result["confidence"] == "high"


def test_detects_eol_and_shutdown_planning():
    result = detect_query_decommissioning_requirement("Need EOL planning and a shutdown plan for this platform.")

    assert result["requires_decommissioning_plan"] is True
    assert result["decommissioning_terms"] == ["end_of_life"]
    assert result["lifecycle_terms"] == ["shutdown"]


def test_non_system_retirement_language_does_not_trigger():
    assert detect_query_decommissioning_requirement("Summarize retirement benefits for employees.") == {
        "requires_decommissioning_plan": False,
        "decommissioning_terms": [],
        "matched_phrases": [],
        "lifecycle_terms": [],
        "recommendations": [],
        "confidence": "none",
    }
