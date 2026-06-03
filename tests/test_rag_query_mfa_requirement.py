from graph.rag.query_mfa_requirement import detect_query_mfa_requirements


def test_detects_common_mfa_factors():
    result = detect_query_mfa_requirements(
        "For user login, require MFA with an authenticator app, hardware security keys, SMS fallback, and recovery codes."
    )

    assert result["has_mfa_requirements"] is True
    assert result["requirements"] == [
        {"category": "authenticator_app", "matched_text": "authenticator app", "severity": "medium"},
        {"category": "hardware_key", "matched_text": "hardware security keys", "severity": "high"},
        {"category": "recovery_codes", "matched_text": "recovery codes", "severity": "medium"},
        {"category": "sms_fallback", "matched_text": "SMS fallback", "severity": "medium"},
    ]


def test_detects_admin_enforcement_and_step_up_authentication():
    result = detect_query_mfa_requirements(
        "Admins must require MFA, and privileged access should use step-up authentication for sensitive actions."
    )

    assert result["has_mfa_requirements"] is True
    assert result["requirements"] == [
        {"category": "admin_enforcement", "matched_text": "Admins must require MFA", "severity": "high"},
        {"category": "step_up_authentication", "matched_text": "step-up authentication", "severity": "high"},
    ]


def test_ignores_unrelated_factor_language_without_authentication_context():
    assert detect_query_mfa_requirements("Compare the factor analysis method for math scoring.") == {
        "has_mfa_requirements": False,
        "requirements": [],
    }
