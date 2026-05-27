from __future__ import annotations

from graph.rag.answer_privacy_leakage import audit_answer_privacy_leakage


def test_detects_email_phone_address_and_ssn_with_line_numbers():
    report = audit_answer_privacy_leakage(
        "Reach Ada at ada.lovelace@example.com or 415-555-0100.\n"
        "Ship records to 123 Main Street and SSN 123-45-6789."
    )

    assert report["has_privacy_leakage_risk"] is True
    assert report["risk_types"] == ["email", "phone", "physical_address", "ssn"]
    assert [sample["line_number"] for sample in report["samples"]] == [1, 1, 2, 2]
    assert all("ada.lovelace@example.com" not in sample["redacted_value"] for sample in report["samples"])
    assert all("123 Main Street" not in sample["redacted_value"] for sample in report["samples"])


def test_detects_api_keys_secret_assignments_credit_cards_and_private_keys():
    answer = """Use token sk-test_1234567890abcdefABCDEF.
password = "correct-horse-battery-staple"
Card: 4111 1111 1111 1111
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASC
-----END PRIVATE KEY-----"""

    report = audit_answer_privacy_leakage(answer)

    assert report["risk_types"] == ["private_key", "api_key", "secret_assignment", "credit_card"]
    samples_by_type = {sample["risk_type"]: sample["redacted_value"] for sample in report["samples"]}
    assert samples_by_type["private_key"] == "[REDACTED PRIVATE KEY]"
    assert "sk-test_1234567890abcdefABCDEF" not in samples_by_type["api_key"]
    assert "correct-horse-battery-staple" not in samples_by_type["secret_assignment"]
    assert samples_by_type["credit_card"] == "***1111"


def test_common_benign_numeric_text_avoids_obvious_false_positives():
    report = audit_answer_privacy_leakage(
        "The 2024 sample had 1,234 rows, a 95% interval, ISBN 978-0-306-40615-7, and ticket 1234567890123."
    )

    assert report == {
        "has_privacy_leakage_risk": False,
        "risk_types": [],
        "samples": [],
    }


def test_clean_answer_returns_negative_report():
    report = audit_answer_privacy_leakage("The public benchmark improved by 4.2 points in the latest run.")

    assert report["has_privacy_leakage_risk"] is False
    assert report["risk_types"] == []
    assert report["samples"] == []
