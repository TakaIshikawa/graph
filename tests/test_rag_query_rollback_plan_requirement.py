from graph.rag.query_rollback_plan_requirement import detect_query_rollback_plan_requirement


def test_deployment_rollback_plan_is_flagged():
    report = detect_query_rollback_plan_requirement("Create a rollback and backout plan for the deployment.")

    assert report["requires_rollback_plan"] is True
    assert report["rollback_terms"] == ["rollback", "backout_plan"]
    assert report["risk_terms"] == ["deployment"]
    assert report["confidence"] == "high"


def test_migration_and_feature_flag_cues_are_categorized():
    report = detect_query_rollback_plan_requirement("Plan migration rollback, recovery checkpoint, and disable the feature flag.")

    assert report["rollback_terms"] == ["feature_flag_disablement", "migration_rollback", "recovery_checkpoint"]
    assert report["risk_terms"] == ["migration", "feature_flag"]
    assert "verify migration recovery point" in report["recommendations"]


def test_generic_undo_wording_does_not_trigger():
    assert detect_query_rollback_plan_requirement("How do I undo a text edit in the UI?") == {
        "requires_rollback_plan": False,
        "rollback_terms": [],
        "matched_phrases": [],
        "risk_terms": [],
        "recommendations": [],
        "confidence": "none",
    }
