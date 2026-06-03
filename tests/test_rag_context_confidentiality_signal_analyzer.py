from __future__ import annotations

from graph.rag.context_confidentiality_signal import analyze_context_confidentiality_signals


def test_detects_internal_confidential_and_nda_context():
    summary = analyze_context_confidentiality_signals(
        [
            {"id": "a", "text": "Internal use only. Confidential NDA roadmap notes; do not share externally."},
        ]
    )

    assert summary["flagged_context_item_count"] == 1
    assert summary["signal_counts"]["internal_only"] == 1
    assert summary["signal_counts"]["confidential"] == 1
    assert summary["signal_counts"]["nda"] == 1
    assert summary["items"][0]["severity"] == "high"
    assert {"signal": "do_not_share", "matched_text": "do not share", "severity": "high"} in summary["items"][0]["findings"]


def test_detects_credential_like_confidentiality_cues():
    summary = analyze_context_confidentiality_signals(
        [
            {"id": "secret", "text": "Customer data export. private key and access token: abcdef1234567890 should rotate."},
        ]
    )

    assert summary["items"][0]["context_id"] == "secret"
    assert summary["items"][0]["severity"] == "critical"
    assert summary["signal_counts"]["customer_data"] == 1
    assert summary["signal_counts"]["private_key"] == 1
    assert summary["signal_counts"]["token"] == 1


def test_benign_public_documentation_is_not_flagged():
    assert analyze_context_confidentiality_signals([{"id": "public", "text": "Public API documentation for tokenization in NLP."}]) == {
        "flagged_context_item_count": 0,
        "finding_count": 0,
        "signal_counts": {
            "confidential": 0,
            "nda": 0,
            "internal_only": 0,
            "customer_data": 0,
            "private_key": 0,
            "token": 0,
            "secret": 0,
            "do_not_share": 0,
        },
        "items": [],
        "confidentiality_review_recommended": False,
    }


def test_mixed_snippets_only_return_flagged_items():
    summary = analyze_context_confidentiality_signals(
        [
            {"id": "public", "text": "Public changelog."},
            {"id": "internal", "metadata": {"note": "Company-internal customer records. client_secret=abcdef123456"}},
        ]
    )

    assert summary["flagged_context_item_count"] == 1
    assert summary["items"][0]["context_id"] == "internal"
    assert {finding["signal"] for finding in summary["items"][0]["findings"]} == {"internal_only", "customer_data", "secret"}
