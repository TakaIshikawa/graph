from graph.rag.query_scim_provisioning_requirement import detect_query_scim_provisioning_requirements


def test_detects_scim_and_automated_user_provisioning():
    result = detect_query_scim_provisioning_requirements(
        "Require SCIM with automated user provisioning for enterprise tenants."
    )

    assert result["has_scim_provisioning_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["automated_provisioning", "scim"]


def test_detects_deprovisioning_group_sync_directory_sync_and_lifecycle():
    result = detect_query_scim_provisioning_requirements(
        "Need deprovisioning, group membership sync, directory sync, and joiner-mover-leaver lifecycle support."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "deprovisioning",
        "directory_sync",
        "group_sync",
        "identity_lifecycle",
    ]


def test_detects_protocol_name_and_provider_directory_phrasing():
    result = detect_query_scim_provisioning_requirements(
        "Support System for Cross-Domain Identity Management plus Okta directory sync."
    )

    assert [row["category"] for row in result["requirements"]] == ["directory_sync", "scim"]


def test_unrelated_login_or_generic_sso_queries_return_defaults():
    assert detect_query_scim_provisioning_requirements("Support SAML SSO and normal login through Okta.") == {
        "has_scim_provisioning_requirements": False,
        "requirements": [],
    }
    assert detect_query_scim_provisioning_requirements("Compare login page UX and password reset copy.") == {
        "has_scim_provisioning_requirements": False,
        "requirements": [],
    }
