from __future__ import annotations

from graph.rag.query_circuit_breaker_requirement import detect_query_circuit_breaker_requirement


def test_detects_circuit_breaker_states_thresholds_and_fallbacks():
    result = detect_query_circuit_breaker_requirement(
        "Document circuit breaker behavior: open state, half-open probes, failure threshold, "
        "fallback path, and dependency isolation."
    )

    assert result["has_circuit_breaker_requirement"] is True
    assert result["requirements"] == [
        {"category": "breaker_state", "matched_text": "open state"},
        {"category": "circuit_breaker", "matched_text": "circuit breaker"},
        {"category": "dependency_isolation", "matched_text": "dependency isolation"},
        {"category": "failure_threshold", "matched_text": "failure threshold"},
        {"category": "fallback_path", "matched_text": "fallback path"},
    ]


def test_electrical_breaker_query_does_not_match():
    assert detect_query_circuit_breaker_requirement(
        "Which electrical breaker size is required for a kitchen outlet?"
    ) == {"has_circuit_breaker_requirement": False, "requirements": []}
