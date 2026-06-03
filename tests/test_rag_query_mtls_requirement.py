from graph.rag.query_mtls_requirement import detect_query_mtls_requirements


def test_detects_mtls_acronym_and_mutual_tls_spelling():
    result = detect_query_mtls_requirements("mTLS and mutual TLS requirements need service-to-service authentication.")

    assert result["has_mtls_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["service_authentication"]


def test_detects_client_certificate_and_ca_trust():
    result = detect_query_mtls_requirements("Mutual TLS clients must present client certificates chained to a trusted CA.")

    assert [row["category"] for row in result["requirements"]] == ["ca_trust", "client_certificate"]


def test_detects_ca_trust_rotation_and_handshake_validation():
    result = detect_query_mtls_requirements("mTLS needs CA trust, certificate rotation, and TLS handshake validation.")

    assert [row["category"] for row in result["requirements"]] == ["ca_trust", "handshake_validation", "rotation"]


def test_ignores_one_way_tls_or_encryption_without_mutual_context():
    assert detect_query_mtls_requirements("Require TLS encryption in transit for browser traffic.") == {
        "has_mtls_requirements": False,
        "requirements": [],
    }
