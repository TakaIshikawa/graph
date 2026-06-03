from graph.rag.query_oidc_requirement import detect_query_oidc_requirements


def test_detects_oidc_acronym_and_openid_connect_spelling():
    result = detect_query_oidc_requirements("OIDC docs must explain the ID token and OpenID Connect issuer.")

    assert result["has_oidc_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["id_token", "issuer_discovery"]


def test_detects_jwks_and_discovery_requirements():
    result = detect_query_oidc_requirements("OpenID Connect metadata should include JWKS URI and discovery document details.")

    assert [row["category"] for row in result["requirements"]] == ["issuer_discovery", "jwks"]


def test_detects_claims_mapping_and_nonce_state_validation():
    result = detect_query_oidc_requirements("For OIDC, require claims mapping plus nonce and state validation.")

    assert [row["category"] for row in result["requirements"]] == ["claims_mapping", "nonce_state"]


def test_ignores_generic_identity_provider_without_oidc_context():
    assert detect_query_oidc_requirements("How should the identity provider map user attributes?") == {
        "has_oidc_requirements": False,
        "requirements": [],
    }
