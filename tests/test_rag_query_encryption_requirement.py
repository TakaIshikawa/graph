from graph.rag.query_encryption_requirement import detect_query_encryption_requirements


def test_detects_encryption_at_rest_and_in_transit_requirements():
    result = detect_query_encryption_requirements("Require encryption at rest for backups and encryption in transit for API traffic.")

    assert result == {
        "has_encryption_requirements": True,
        "requirements": [
            {"category": "encryption_at_rest", "matched_text": "encryption at rest", "severity": "high"},
            {"category": "encryption_in_transit", "matched_text": "encryption in transit", "severity": "high"},
        ],
    }


def test_detects_key_management_rotation_and_customer_managed_keys():
    result = detect_query_encryption_requirements("Use KMS with customer-managed keys and scheduled key rotation for tenant secrets.")

    assert result["has_encryption_requirements"] is True
    assert result["requirements"] == [
        {"category": "customer_managed_keys", "matched_text": "customer-managed keys", "severity": "high"},
        {"category": "key_rotation", "matched_text": "scheduled key rotation", "severity": "medium"},
        {"category": "kms", "matched_text": "KMS", "severity": "medium"},
    ]


def test_detects_transport_security_wording():
    result = detect_query_encryption_requirements("Document TLS 1.3 handling and HTTPS only requirements for clients.")

    assert result["requirements"] == [{"category": "tls", "matched_text": "TLS 1.3", "severity": "high"}]


def test_unrelated_query_returns_empty_results():
    assert detect_query_encryption_requirements("Compare database indexing strategies for reporting dashboards.") == {
        "has_encryption_requirements": False,
        "requirements": [],
    }
