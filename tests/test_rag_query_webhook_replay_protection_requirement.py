from __future__ import annotations

from graph.rag.query_webhook_replay_protection_requirement import (
    detect_query_webhook_replay_protection_requirement,
)


def test_detects_webhook_replay_protection_categories():
    result = detect_query_webhook_replay_protection_requirement(
        "Webhook replay protection must use timestamp tolerance, a nonce, signature timestamp, "
        "duplicate delivery rejection, and a replay window."
    )

    assert result["has_webhook_replay_protection_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "duplicate_delivery_rejection",
        "nonce",
        "replay_protection",
        "replay_window",
        "signature_timestamp",
        "timestamp_tolerance",
    ]


def test_detects_alternate_replay_wording():
    result = detect_query_webhook_replay_protection_requirement(
        "For webhooks, prevent replay attacks with clock skew handling and deduplicate deliveries."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "duplicate_delivery_rejection",
        "replay_protection",
        "timestamp_tolerance",
    ]


def test_generic_webhook_delivery_without_replay_cues_is_negative():
    assert detect_query_webhook_replay_protection_requirement("Document webhook delivery retries and event payloads.") == {
        "has_webhook_replay_protection_requirement": False,
        "requirements": [],
    }
