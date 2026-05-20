from __future__ import annotations

from graph.rag.context_privacy_redaction import plan_context_privacy_redactions


def test_plans_redactions_with_masked_previews_only():
    report = plan_context_privacy_redactions(
        [
            {
                "id": "r1",
                "text": "Contact ada@example.com or 415-555-0100. Token sk-test1234567890abcdef and card 4111 1111 1111 1111.",
            }
        ]
    )

    assert report["finding_count"] == 4
    assert report["redaction_type_counts"]["email"] == 1
    assert report["redaction_type_counts"]["phone"] == 1
    assert report["redaction_type_counts"]["api_key"] == 1
    assert report["redaction_type_counts"]["credit_card"] == 1
    assert all("example.com" not in finding["masked_preview"] for finding in report["results"][0]["findings"])
    assert "sensitive_context_detected" in report["warnings"]


def test_precise_location_is_low_confidence_opt_in():
    skipped = plan_context_privacy_redactions([{"id": "r1", "text": "Meet at 123 Main Street tomorrow."}])
    included = plan_context_privacy_redactions([{"id": "r1", "text": "Meet at 123 Main Street tomorrow."}], include_low_confidence=True)

    assert skipped["finding_count"] == 0
    assert included["redaction_type_counts"]["precise_location"] == 1
