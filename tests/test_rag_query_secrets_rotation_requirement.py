from __future__ import annotations

from graph.rag import detect_query_secrets_rotation_requirement
from graph.rag.query_secrets_rotation_requirement import detect_query_secrets_rotation_requirements


def test_detects_secret_rotation_requirements_and_cadence_in_order():
    rows = detect_query_secrets_rotation_requirement(
        "Document secret rotation every 90 days, automated API key rotation, credential rotation, password rotation, and token rotation."
    )

    assert [row["category"] for row in rows] == [
        "secret_rotation",
        "automated_rotation",
        "api_key_rotation",
        "credential_rotation",
        "password_rotation",
        "token_rotation",
    ]
    assert rows[0]["matched_text"] == "secret rotation"
    assert {"matched_text": "every 90 days", "span": [25, 38]} in rows[0]["cadence_cues"]
    assert {"matched_text": "automated API key rotation", "span": [40, 66]} in rows[0]["cadence_cues"]


def test_detects_expiring_credentials():
    rows = detect_query_secrets_rotation_requirements("Which controls cover expiring credentials after 30 days?")

    assert rows[0]["category"] == "expiring_credentials"
    assert rows[0]["matched_text"] == "expiring credentials"
    assert rows[0]["cadence_cues"][0]["matched_text"] == "expiring credentials after 30 days"


def test_unrelated_authentication_setup_is_not_flagged():
    assert detect_query_secrets_rotation_requirement("How do users set up MFA and SSO authentication?") == []
