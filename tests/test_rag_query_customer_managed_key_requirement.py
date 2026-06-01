from graph.rag.query_customer_managed_key_requirement import detect_query_customer_managed_key_requirements


def test_cmk_byok_and_ekm_wording_are_detected():
    report = detect_query_customer_managed_key_requirements(
        "Support CMK, bring your own key, and external key management."
    )

    assert report == {
        "has_customer_managed_key_requirements": True,
        "requirements": ["cmk", "byok", "ekm"],
        "external_key_sensitive": True,
    }


def test_key_custody_and_tenant_key_are_detected():
    report = detect_query_customer_managed_key_requirements(
        "Document customer-managed key custody and tenant-specific keys."
    )

    assert report["requirements"] == ["cmk", "key_custody", "tenant_key"]
    assert report["external_key_sensitive"] is False


def test_unrelated_query_has_no_customer_managed_key_requirements():
    assert detect_query_customer_managed_key_requirements("Rotate application secrets monthly.") == {
        "has_customer_managed_key_requirements": False,
        "requirements": [],
        "external_key_sensitive": False,
    }
