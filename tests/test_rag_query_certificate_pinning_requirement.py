from graph.rag.query_certificate_pinning_requirement import detect_query_certificate_pinning_requirements


def test_detects_certificate_and_public_key_pinning_variants():
    result = detect_query_certificate_pinning_requirements("Require certificate pinning and SPKI pinning for the API client.")

    assert result["has_certificate_pinning_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["certificate_pinning", "public_key_pinning"]
    assert result["rotation_sensitive"] is False


def test_detects_hpkp_mobile_and_bypass_terms():
    result = detect_query_certificate_pinning_requirements("Document HPKP, Android certificate pinning, and pinning bypass controls.")

    assert [row["category"] for row in result["requirements"]] == ["certificate_pinning", "hpkp", "mobile_pinning", "pinning_bypass"]


def test_rotation_or_backup_pins_set_rotation_sensitive():
    result = detect_query_certificate_pinning_requirements("How do we rotate pins and configure backup pins?")

    assert [row["category"] for row in result["requirements"]] == ["pin_rotation", "backup_pins"]
    assert result["rotation_sensitive"] is True


def test_unrelated_certificate_query_returns_defaults():
    assert detect_query_certificate_pinning_requirements("When does the TLS certificate expire?") == {
        "has_certificate_pinning_requirements": False,
        "requirements": [],
        "rotation_sensitive": False,
    }
