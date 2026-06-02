from graph.rag.query_key_management_requirement import detect_query_key_management_requirement


def test_key_management_categories_are_detected():
    report = detect_query_key_management_requirement(
        "Require KMS, key custody, HSM, envelope encryption, key rotation, and key destruction."
    )

    assert report["requires_key_management"] is True
    assert report["categories"] == ["lifecycle", "custody", "hardware_security_module", "envelope_encryption"]
    assert report["matches"][0]["matched_text"] == "KMS"


def test_general_key_management_language_triggers_lifecycle():
    report = detect_query_key_management_requirement("Describe key management controls.")

    assert report["categories"] == ["lifecycle"]


def test_unrelated_cmk_only_query_returns_empty_result():
    assert detect_query_key_management_requirement("Compare customer-managed-key billing options.") == {
        "requires_key_management": False,
        "categories": [],
        "matches": [],
    }
