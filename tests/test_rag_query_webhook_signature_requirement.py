from graph.rag.query_webhook_signature_requirement import detect_query_webhook_signature_requirements


def test_detects_core_webhook_signature_categories():
    result = detect_query_webhook_signature_requirements("For webhooks, verify HMAC with the signing secret and X-Hub-Signature header.")

    assert result["has_webhook_signature_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["hmac_signature", "signature_header", "signing_secret"]


def test_detects_replay_timestamp_and_payload_canonicalization():
    result = detect_query_webhook_signature_requirements("Webhook replay protection needs timestamp tolerance and canonical payload rules.")

    assert [row["category"] for row in result["requirements"]] == ["canonical_payload", "replay_protection", "timestamp_tolerance"]


def test_avoids_generic_signature_without_webhook_context():
    assert detect_query_webhook_signature_requirements("Verify the document signature with a timestamp.") == {
        "has_webhook_signature_requirements": False,
        "requirements": [],
    }
