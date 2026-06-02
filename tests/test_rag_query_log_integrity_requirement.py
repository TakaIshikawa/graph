from graph.rag.query_log_integrity_requirement import detect_query_log_integrity_requirement


def test_log_integrity_categories_are_detected():
    report = detect_query_log_integrity_requirement(
        "Require tamper-evident logs, immutable logs, log signing, audit log integrity, WORM storage, and log chain verification."
    )

    assert report["requires_log_integrity"] is True
    assert report["categories"] == ["tamper_evidence", "immutable_storage", "signing", "chain_verification"]
    assert report["matches"][0]["span"] == (8, 27)


def test_signed_logs_and_worm_storage_are_detected():
    report = detect_query_log_integrity_requirement("Use WORM storage for signed logs.")

    assert report["categories"] == ["immutable_storage", "signing"]


def test_generic_audit_logging_does_not_trigger():
    assert detect_query_log_integrity_requirement("Enable audit logging for admin events.") == {
        "requires_log_integrity": False,
        "categories": [],
        "matches": [],
    }
