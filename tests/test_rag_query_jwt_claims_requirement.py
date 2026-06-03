from graph.rag.query_jwt_claims_requirement import detect_query_jwt_claims_requirements


def test_detects_jwt_registered_claim_names():
    result = detect_query_jwt_claims_requirements("JWT validation must check iss, aud, sub, and exp claims.")

    assert result["has_jwt_claims_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["audience", "expiration", "issuer", "subject"]


def test_detects_scopes_roles_and_custom_claims():
    result = detect_query_jwt_claims_requirements("Access token claims should include scopes, roles claims, and custom claims.")

    assert [row["category"] for row in result["requirements"]] == ["custom_claims", "scopes_roles"]


def test_detects_json_web_token_spelling():
    result = detect_query_jwt_claims_requirements("JSON Web Token requirements include issuer and audience.")

    assert [row["category"] for row in result["requirements"]] == ["audience", "issuer"]


def test_ignores_generic_claim_wording_without_token_context():
    assert detect_query_jwt_claims_requirements("The vendor claims support for custom reporting roles.") == {
        "has_jwt_claims_requirements": False,
        "requirements": [],
    }
