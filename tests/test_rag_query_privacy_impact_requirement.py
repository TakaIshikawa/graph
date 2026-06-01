from graph.rag.query_privacy_impact_requirement import detect_query_privacy_impact_requirements


def test_dpia_pia_and_privacy_impact_wording_are_detected_once():
    report = detect_query_privacy_impact_requirements(
        "Run a DPIA, PIA, and privacy impact assessment before launch."
    )

    assert report == {
        "has_privacy_impact_requirements": True,
        "requirements": ["dpia", "pia", "privacy_impact_assessment"],
        "high_risk_processing_sensitive": False,
    }


def test_data_protection_privacy_review_and_high_risk_processing_are_detected():
    report = detect_query_privacy_impact_requirements(
        "Need data protection impact checks, a privacy review, and high-risk processing controls."
    )

    assert report["requirements"] == ["data_protection_impact", "privacy_review", "high_risk_processing"]
    assert report["high_risk_processing_sensitive"] is True


def test_unrelated_query_has_no_privacy_impact_requirements():
    assert detect_query_privacy_impact_requirements("Summarize privacy policy update dates.") == {
        "has_privacy_impact_requirements": False,
        "requirements": [],
        "high_risk_processing_sensitive": False,
    }
