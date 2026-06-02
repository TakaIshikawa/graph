from graph.rag.query_secrets_management_requirement import detect_query_secrets_management_requirement


def test_detects_credential_rotation_scanning_and_storage():
    result = detect_query_secrets_management_requirement(
        "Cover secret storage in a vault, API key rotation, service account credentials, environment variables, secret scanning, tokens, and least-privilege access."
    )

    assert result["has_secrets_management_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "access_control",
        "api_keys",
        "env_vars",
        "rotation",
        "scanning",
        "service_account",
        "storage",
        "tokens",
        "vault",
    ]


def test_api_key_and_token_need_security_context():
    assert detect_query_secrets_management_requirement("Compare pricing for API tokens and usage tokens.") == {
        "has_secrets_management_requirement": False,
        "requirements": [],
    }

    result = detect_query_secrets_management_requirement("Security controls for storing API keys and rotating tokens.")

    assert [row["category"] for row in result["requirements"]] == ["api_keys", "rotation", "storage", "tokens"]


def test_unrelated_privacy_mentions_are_not_flagged():
    assert detect_query_secrets_management_requirement("Summarize the privacy notice and user consent wording.") == {
        "has_secrets_management_requirement": False,
        "requirements": [],
    }
