from graph.rag.query_vendor_risk_requirement import detect_query_vendor_risk_requirements


def test_vendor_supplier_and_third_party_wording_are_normalized():
    report = detect_query_vendor_risk_requirements(
        "Assess vendor risk, third-party risk, and supplier assessment evidence."
    )

    assert report == {
        "has_vendor_risk_requirements": True,
        "requirements": ["vendor_risk", "third_party_risk", "supplier_assessment"],
        "third_party_sensitive": True,
    }


def test_soc2_questionnaire_subprocessor_and_due_diligence_are_detected():
    report = detect_query_vendor_risk_requirements(
        "Review the SOC 2, collect a security questionnaire, list subprocessors, and confirm due diligence."
    )

    assert report["requirements"] == ["soc2_review", "security_questionnaire", "subprocessors", "due_diligence"]
    assert report["third_party_sensitive"] is True


def test_unrelated_query_has_no_vendor_risk_requirements():
    assert detect_query_vendor_risk_requirements("Compare vendor pricing and renewal dates.") == {
        "has_vendor_risk_requirements": False,
        "requirements": [],
        "third_party_sensitive": False,
    }
