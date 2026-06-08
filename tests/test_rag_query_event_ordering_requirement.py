from __future__ import annotations

from graph.rag.query_event_ordering_requirement import detect_query_event_ordering_requirement


def test_detects_event_ordering_categories_and_text():
    result = detect_query_event_ordering_requirement(
        "Need event ordering guarantees for FIFO queue delivery, sequence numbers, "
        "out-of-order delivery handling, and partition ordering."
    )

    assert result["has_event_ordering_requirement"] is True
    assert result["requirements"] == [
        {"category": "event_ordering", "matched_text": "event ordering"},
        {"category": "fifo_queue", "matched_text": "FIFO queue"},
        {"category": "out_of_order_delivery", "matched_text": "out-of-order delivery"},
        {"category": "partition_ordering", "matched_text": "partition ordering"},
        {"category": "sequence_number", "matched_text": "sequence numbers"},
    ]


def test_ignores_generic_event_wording():
    assert detect_query_event_ordering_requirement(
        "List event venues in alphabetical order."
    ) == {"has_event_ordering_requirement": False, "requirements": []}
