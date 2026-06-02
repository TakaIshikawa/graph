from graph.rag.query_business_continuity_requirement import detect_query_business_continuity_requirement


def test_business_continuity_categories_are_detected():
    report = detect_query_business_continuity_requirement(
        "Require business continuity, BCP testing, crisis management, alternate worksite, RTO, and continuity plan owner."
    )

    assert report["requires_business_continuity"] is True
    assert report["categories"] == [
        "continuity_plan",
        "testing",
        "crisis_management",
        "alternate_worksite",
        "recovery_objective",
        "ownership",
    ]


def test_continuity_testing_and_recovery_objective_variants_are_detected():
    report = detect_query_business_continuity_requirement("Continuity testing must validate recovery time objective.")

    assert report["categories"] == ["testing", "recovery_objective"]


def test_disaster_recovery_only_wording_does_not_trigger():
    assert detect_query_business_continuity_requirement("Review the disaster recovery plan.") == {
        "requires_business_continuity": False,
        "categories": [],
        "matches": [],
    }
