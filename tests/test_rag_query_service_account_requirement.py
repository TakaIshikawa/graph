from graph.rag.query_service_account_requirement import detect_query_service_account_requirement


def test_service_account_machine_user_and_workload_identity_trigger_detector():
    result = detect_query_service_account_requirement("Document service accounts, machine users, and workload identity requirements.")

    assert result["requires_service_account"] is True
    assert result["identity_types"] == ["machine_user", "service_account", "workload_identity"]
    assert result["severity"] == "medium"


def test_rotation_mentions_are_detected_for_non_human_identity():
    result = detect_query_service_account_requirement("Non-human identity access needs a credential rotation schedule.")

    assert result["requires_service_account"] is True
    assert result["identity_types"] == ["non_human_identity"]
    assert result["rotation_mentioned"] is True
    assert result["severity"] == "high"


def test_human_login_query_returns_defaults_even_with_rotation_word():
    assert detect_query_service_account_requirement("Human users should rotate passwords after normal login.") == {
        "requires_service_account": False,
        "identity_types": [],
        "rotation_mentioned": False,
        "matched_cues": [],
        "severity": "none",
    }
