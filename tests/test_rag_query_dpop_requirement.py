from graph.rag.query_dpop_requirement import detect_dpop_requirement


def test_detects_dpop_specific_requirements():
    result = detect_dpop_requirement("OAuth DPoP requirements must include the DPoP proof JWT and jkt thumbprint.")

    assert result["has_dpop_requirement"] is True
    assert result["requirements"] == [
        {"category": "dpop", "matched_text": "DPoP", "trigger_type": "dpop_specific", "severity": "high"},
        {"category": "dpop_proof_jwt", "matched_text": "DPoP proof JWT", "trigger_type": "dpop_specific", "severity": "high"},
        {"category": "jkt_thumbprint", "matched_text": "jkt thumbprint", "trigger_type": "dpop_specific", "severity": "high"},
    ]


def test_detects_proof_of_possession_and_sender_constrained_tokens():
    result = detect_dpop_requirement("Require proof-of-possession token support and sender-constrained tokens.")

    assert [row["category"] for row in result["requirements"]] == [
        "proof_of_possession_token",
        "sender_constrained_token",
    ]
    assert {row["trigger_type"] for row in result["requirements"]} == {"proof_of_possession"}


def test_detects_mixed_case_token_binding():
    result = detect_dpop_requirement("OAuth should support ToKeN BiNdInG for bound access tokens.")

    assert result["requirements"] == [
        {"category": "token_binding", "matched_text": "ToKeN BiNdInG", "trigger_type": "proof_of_possession", "severity": "medium"}
    ]


def test_ignores_unrelated_jwt_claim_queries():
    assert detect_dpop_requirement("JWT validation must check issuer, audience, subject, and expiration claims.") == {
        "has_dpop_requirement": False,
        "requirements": [],
    }


def test_jwt_query_with_dpop_proof_language_triggers():
    result = detect_dpop_requirement("JWT docs must describe proof-of-possession token requirements.")

    assert result["has_dpop_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == ["proof_of_possession_token"]
