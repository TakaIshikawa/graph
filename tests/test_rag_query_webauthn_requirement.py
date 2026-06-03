from graph.rag.query_webauthn_requirement import detect_query_webauthn_requirements


def test_detects_webauthn_and_fido2_terms():
    result = detect_query_webauthn_requirements("WebAuthn and FIDO2 requirements should cover authenticator attestation.")

    assert result["has_webauthn_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["attestation"]


def test_detects_passkey_and_user_verification():
    result = detect_query_webauthn_requirements("Passkey rollout must require user verification.")

    assert [row["category"] for row in result["requirements"]] == ["passkey", "user_verification"]


def test_detects_relying_party_id_resident_key_and_challenge():
    result = detect_query_webauthn_requirements("WebAuthn needs relying party ID, resident keys, and challenge handling.")

    assert [row["category"] for row in result["requirements"]] == ["challenge", "relying_party", "resident_key"]


def test_ignores_generic_web_authentication_without_webauthn_context():
    assert detect_query_webauthn_requirements("How should web authentication handle password reset challenges?") == {
        "has_webauthn_requirements": False,
        "requirements": [],
    }
