from __future__ import annotations

from graph.rag.query_dead_letter_queue_requirement import detect_query_dead_letter_queue_requirement


def test_detects_dlq_poison_retry_quarantine_and_redrive():
    result = detect_query_dead_letter_queue_requirement(
        "Need DLQ handling for poison messages, retry exhaustion, quarantine queue, and redrive policy."
    )

    assert result["has_dead_letter_queue_requirement"] is True
    assert result["requirements"] == [
        {"category": "dead_letter_queue", "matched_text": "DLQ"},
        {"category": "poison_message", "matched_text": "poison messages"},
        {"category": "quarantine_queue", "matched_text": "quarantine queue"},
        {"category": "redrive_policy", "matched_text": "redrive policy"},
        {"category": "retry_exhaustion", "matched_text": "retry exhaustion"},
    ]


def test_detects_spelled_out_dead_letter_queue():
    result = detect_query_dead_letter_queue_requirement("Explain dead letter queue retention.")

    assert result["requirements"] == [{"category": "dead_letter_queue", "matched_text": "dead letter queue"}]


def test_ignores_non_queue_dead_letter_and_generic_retry():
    assert detect_query_dead_letter_queue_requirement(
        "The office sends dead letters by mail and users retry login after failure."
    ) == {"has_dead_letter_queue_requirement": False, "requirements": []}
