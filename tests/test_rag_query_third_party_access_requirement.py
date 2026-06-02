from graph.rag.query_third_party_access_requirement import detect_query_third_party_access_requirement


def test_third_party_access_categories_are_detected():
    report = detect_query_third_party_access_requirement(
        "Require vendor access, contractor access, support engineer access, external admin access, and JIT third-party access."
    )

    assert report["requires_third_party_access"] is True
    assert report["categories"] == [
        "vendor_access",
        "contractor_access",
        "support_access",
        "external_admin",
        "just_in_time_access",
    ]


def test_just_in_time_access_variant_is_detected():
    report = detect_query_third_party_access_requirement("Use just-in-time third-party access.")

    assert report["categories"] == ["just_in_time_access", "vendor_access"]


def test_generic_vendor_risk_does_not_trigger():
    assert detect_query_third_party_access_requirement("Assess third-party vendor risk before onboarding.") == {
        "requires_third_party_access": False,
        "categories": [],
        "matches": [],
    }
