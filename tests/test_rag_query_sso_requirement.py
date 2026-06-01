from graph.rag.query_sso_requirement import detect_query_sso_requirement


def test_extracts_known_sso_providers_in_deterministic_order():
    result = detect_query_sso_requirement("Allow Okta, Google Workspace login, and Azure AD SSO.")

    assert result["requires_sso"] is True
    assert result["providers"] == ["azure_ad", "google_workspace", "okta"]
    assert result["confidence"] == "high"


def test_extracts_sso_protocols_and_identity_provider_cues():
    result = detect_query_sso_requirement("Support SAML 2.0, OIDC, and login through an identity provider.")

    assert result["requires_sso"] is True
    assert result["protocols"] == ["oidc", "saml"]
    assert [cue["category"] for cue in result["matched_cues"]] == ["identity_provider"]
    assert result["confidence"] == "high"


def test_non_authentication_query_returns_defaults():
    assert detect_query_sso_requirement("Summarize workspace storage limits.") == {
        "requires_sso": False,
        "providers": [],
        "protocols": [],
        "matched_cues": [],
        "confidence": "none",
    }
