from __future__ import annotations

from graph.rag.query_encryption_key_management_requirement import detect_query_encryption_key_management_requirements


def test_detects_encryption_key_management_categories_and_severities():
    result = detect_query_encryption_key_management_requirements(
        "For cloud security, require customer-managed keys, KMS/HSM backing, key rotation, "
        "envelope encryption, key escrow, and BYOK support."
    )

    assert result["has_encryption_key_management_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "bring_your_own_key",
        "customer_managed_key",
        "hsm_kms",
        "envelope_encryption",
        "key_escrow",
        "key_rotation",
    ]
    high_categories = {row["category"] for row in result["requirements"] if row["severity"] == "high"}
    assert {"customer_managed_key", "hsm_kms", "bring_your_own_key"} <= high_categories


def test_detects_contextual_key_rotation_and_escrow_requirements():
    result = detect_query_encryption_key_management_requirements(
        "Find data protection docs that explain encryption key rotation and key escrow."
    )

    assert result["requirements"] == [
        {"category": "key_escrow", "severity": "medium", "matched_text": "key escrow", "span": (67, 77)},
        {"category": "key_rotation", "severity": "medium", "matched_text": "key rotation", "span": (50, 62)},
    ]


def test_physical_key_language_without_security_context_is_ignored():
    result = detect_query_encryption_key_management_requirements(
        "What is the office key rotation schedule for physical keys and key cards?"
    )

    assert result == {
        "has_encryption_key_management_requirements": False,
        "requirements": [],
    }


def test_kms_and_customer_managed_mentions_are_high_severity_without_extra_context():
    result = detect_query_encryption_key_management_requirements("Compare BYOK, CMK, and KMS options.")

    assert result["has_encryption_key_management_requirements"] is True
    assert [row["severity"] for row in result["requirements"]] == ["high", "high", "high"]
