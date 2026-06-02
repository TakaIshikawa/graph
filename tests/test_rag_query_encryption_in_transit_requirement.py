import pytest

from graph.rag.query_encryption_in_transit_requirement import detect_query_encryption_in_transit_requirement


def test_detects_transport_encryption_values_and_cues():
    result = detect_query_encryption_in_transit_requirement(
        "Require HTTPS, mTLS, minimum TLS 1.2 or TLS 1.3, certificate validation, forward secrecy, and TLS_AES_128_GCM_SHA256 cipher suites."
    )

    assert result["requires_encryption_in_transit"] is True
    assert result["cue_categories"] == [
        "tls",
        "https",
        "mtls",
        "certificate_validation",
        "cipher_suite",
        "forward_secrecy",
        "minimum_tls_version",
    ]
    assert result["protocol_values"] == ["HTTPS", "mTLS", "TLS 1.2", "TLS 1.3", "TLS_AES_128_GCM_SHA256"]


def test_at_rest_only_encryption_does_not_trigger_transport_requirement():
    assert detect_query_encryption_in_transit_requirement("Describe database encryption at rest.") == {
        "requires_encryption_in_transit": False,
        "cue_categories": [],
        "protocol_values": [],
    }


@pytest.mark.parametrize("query", ["", " ", None])
def test_invalid_query_raises_value_error(query):
    with pytest.raises(ValueError):
        detect_query_encryption_in_transit_requirement(query)  # type: ignore[arg-type]
