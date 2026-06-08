from __future__ import annotations

from graph.rag.query_exactly_once_delivery_requirement import detect_query_exactly_once_delivery_requirement


def test_detects_delivery_semantics_categories():
    result = detect_query_exactly_once_delivery_requirement(
        "Compare exactly-once, at-least-once, and at-most-once delivery semantics with a deduplication window."
    )

    assert result["has_exactly_once_delivery_requirement"] is True
    assert result["requirements"] == [
        {"category": "at_least_once", "matched_text": "at-least-once"},
        {"category": "at_most_once", "matched_text": "at-most-once"},
        {"category": "deduplication_window", "matched_text": "deduplication window"},
        {"category": "delivery_semantics", "matched_text": "delivery semantics"},
        {"category": "exactly_once", "matched_text": "exactly-once"},
    ]


def test_ignores_shipping_and_generic_notification_text():
    assert detect_query_exactly_once_delivery_requirement(
        "Track package delivery once it ships and send one notification email."
    ) == {"has_exactly_once_delivery_requirement": False, "requirements": []}
