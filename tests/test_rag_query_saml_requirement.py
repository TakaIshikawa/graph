from graph.rag.query_saml_requirement import detect_query_saml_requirements


def test_detects_saml_metadata_and_name_id_format():
    result = detect_query_saml_requirements(
        "Need SAML SSO with IdP metadata XML upload and NameID format configuration."
    )

    assert result["has_saml_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["metadata_xml", "name_id_format"]


def test_detects_assertion_signing_attribute_mapping_and_flows():
    result = detect_query_saml_requirements(
        "Support identity provider SAML assertion signing, map attributes, IdP-initiated login, and SP initiated login."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "assertion_signing",
        "attribute_mapping",
        "idp_initiated_login",
        "sp_initiated_login",
    ]


def test_detects_certificate_rotation_with_identity_context():
    result = detect_query_saml_requirements("For SSO, require certificate rotation and certificate rollover documentation.")

    assert [row["category"] for row in result["requirements"]] == ["certificate_rotation"]


def test_ignores_generic_xml_or_signing_without_identity_context():
    assert detect_query_saml_requirements("Validate XML signature examples and signed documents.") == {
        "has_saml_requirements": False,
        "requirements": [],
    }
    assert detect_query_saml_requirements("How should generic metadata XML be structured for a config file?") == {
        "has_saml_requirements": False,
        "requirements": [],
    }
