from graph.rag.query_sso_provisioning_requirement import detect_query_sso_provisioning_requirement


def test_detects_jit_and_automatic_account_provisioning():
    result = detect_query_sso_provisioning_requirement(
        "Need JIT provisioning and automatic account provisioning for SSO users."
    )

    assert result["has_sso_provisioning_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "automatic_account_provisioning",
        "just_in_time_provisioning",
    ]
    assert {row["severity"] for row in result["requirements"]} == {"high"}


def test_detects_idp_initiated_user_creation():
    result = detect_query_sso_provisioning_requirement(
        "Compare IdP-initiated user creation and creating accounts after single sign-on login."
    )

    assert [row["category"] for row in result["requirements"]] == ["idp_initiated_user_creation"]
    assert result["requirements"][0]["matched_text"] == "IdP-initiated user creation"


def test_detects_deprovisioning_and_account_lifecycle_language():
    result = detect_query_sso_provisioning_requirement(
        "Require deprovisioning, account lifecycle controls, and joiner-mover-leaver workflow evidence."
    )

    assert [row["category"] for row in result["requirements"]] == ["account_lifecycle", "deprovisioning"]
    assert result["requirements"][0]["severity"] == "medium"
    assert result["requirements"][1]["severity"] == "high"


def test_plain_sso_queries_without_lifecycle_intent_remain_negative():
    assert detect_query_sso_provisioning_requirement("Support SAML SSO with Okta and Azure AD.") == {
        "has_sso_provisioning_requirement": False,
        "requirements": [],
    }

