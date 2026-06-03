from graph.rag.query_pkce_requirement import detect_query_pkce_requirements


def test_detects_pkce_acronym_and_challenge_verifier():
    result = detect_query_pkce_requirements("PKCE docs must define the code challenge and code verifier.")

    assert result["has_pkce_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["code_challenge", "code_verifier"]


def test_detects_s256_and_public_client_context():
    result = detect_query_pkce_requirements("OAuth public clients must use S256 for PKCE.")

    assert [row["category"] for row in result["requirements"]] == ["public_client", "s256_method"]


def test_detects_native_and_mobile_oauth_flow():
    result = detect_query_pkce_requirements("Authorization code flow for native apps and mobile applications should require PKCE.")

    assert [row["category"] for row in result["requirements"]] == ["native_app_flow"]


def test_ignores_generic_challenge_verifier_without_oauth_context():
    assert detect_query_pkce_requirements("Create a puzzle challenge and verifier for the training exercise.") == {
        "has_pkce_requirements": False,
        "requirements": [],
    }
