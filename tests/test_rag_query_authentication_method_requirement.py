from __future__ import annotations

from graph.rag.query_authentication_method_requirement import detect_query_authentication_method_requirements


def test_detects_authentication_method_requirements_in_position_order():
    rows = detect_query_authentication_method_requirements("Require passkeys, MFA, SSO via OIDC, SAML, API keys, and service accounts.")

    assert [row["category"] for row in rows] == [
        "passwordless_passkey",
        "mfa",
        "sso",
        "oauth_oidc",
        "saml",
        "api_key",
        "service_account",
    ]
    assert rows[0]["matched_text"] == "passkeys"
    assert rows[0]["span"] == [8, 16]


def test_unrelated_authentication_method_query_returns_empty_list():
    assert detect_query_authentication_method_requirements("Compare support response times.") == []
