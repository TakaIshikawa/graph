from graph.rag.query_oauth_authorization_code_requirement import detect_query_oauth_authorization_code_requirements


def test_detects_multiple_oauth_authorization_code_requirement_categories():
    result = detect_query_oauth_authorization_code_requirements(
        "OAuth 2.0 authorization code flow must document the authorization endpoint, token endpoint, "
        "client secret handling, and authorization code exchange."
    )

    assert result["has_oauth_authorization_code_requirements"] is True
    assert result["requirements"] == [
        {"category": "authorization_code_exchange", "matched_text": "authorization code exchange", "severity": "high"},
        {"category": "authorization_endpoint", "matched_text": "authorization endpoint", "severity": "high"},
        {"category": "client_secret", "matched_text": "client secret", "severity": "high"},
        {"category": "token_endpoint", "matched_text": "token endpoint", "severity": "high"},
    ]


def test_detects_redirect_uri_wording():
    result = detect_query_oauth_authorization_code_requirements(
        "For the OAuth authorization code grant, require exact redirect URI registration and callback URI validation."
    )

    assert result["has_oauth_authorization_code_requirements"] is True
    assert result["requirements"] == [
        {"category": "redirect_uri", "matched_text": "redirect URI", "severity": "high"},
    ]


def test_detects_token_endpoint_wording():
    result = detect_query_oauth_authorization_code_requirements(
        "OIDC code flow docs should include the token endpoint used by confidential clients."
    )

    assert result["has_oauth_authorization_code_requirements"] is True
    assert result["requirements"] == [
        {"category": "client_secret", "matched_text": "confidential clients", "severity": "high"},
        {"category": "token_endpoint", "matched_text": "token endpoint", "severity": "high"},
    ]


def test_ignores_generic_authorization_questions_without_oauth_or_code_flow_context():
    assert detect_query_oauth_authorization_code_requirements("Which authorization checks protect this admin endpoint?") == {
        "has_oauth_authorization_code_requirements": False,
        "requirements": [],
    }
