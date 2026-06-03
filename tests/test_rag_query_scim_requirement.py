from graph.rag.query_scim_requirement import detect_scim_requirement


def test_scim_and_user_provisioning_queries_trigger_requirement():
    result = detect_scim_requirement("Require SCIM with automated user provisioning for enterprise tenants.")

    assert result["requires_scim"] is True
    assert result["categories"] == ["scim", "user_provisioning"]
    assert result["matches"][0]["matched_text"] == "SCIM"
    assert "SCIM with automated user provisioning" in result["matches"][0]["snippet"]


def test_detects_deprovisioning_group_sync_and_directory_lifecycle_language():
    result = detect_scim_requirement(
        "Need deprovisioning, group membership sync, directory sync, and identity lifecycle controls."
    )

    assert result["categories"] == [
        "deprovisioning",
        "group_sync",
        "directory_sync",
        "identity_lifecycle",
    ]


def test_detects_group_provisioning_and_protocol_name_case_insensitively():
    result = detect_scim_requirement(
        "Support SYSTEM FOR CROSS-DOMAIN IDENTITY MANAGEMENT plus group provisioning."
    )

    assert result["categories"] == ["scim", "group_provisioning"]
    assert result["matches"][0]["matched_text"] == "SYSTEM FOR CROSS-DOMAIN IDENTITY MANAGEMENT"
    assert result["matches"][0]["span"] == (8, 51)


def test_generic_account_management_without_provisioning_intent_is_negative():
    assert detect_scim_requirement("Compare account management screens and profile settings for end users.") == {
        "requires_scim": False,
        "categories": [],
        "matches": [],
    }
