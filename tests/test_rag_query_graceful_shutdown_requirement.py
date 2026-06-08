from __future__ import annotations

from graph.rag.query_graceful_shutdown_requirement import detect_query_graceful_shutdown_requirement


def test_detects_service_graceful_shutdown_categories():
    result = detect_query_graceful_shutdown_requirement(
        "Service needs graceful shutdown, drain connections, termination grace period, "
        "complete in-flight requests, SIGTERM handling, and shutdown hook docs."
    )

    assert result["has_graceful_shutdown_requirement"] is True
    assert result["requirements"] == [
        {"category": "connection_drain", "matched_text": "drain connections"},
        {"category": "graceful_shutdown", "matched_text": "graceful shutdown"},
        {"category": "inflight_completion", "matched_text": "complete in-flight requests"},
        {"category": "shutdown_hook", "matched_text": "shutdown hook"},
        {"category": "sigterm_handling", "matched_text": "SIGTERM"},
        {"category": "termination_grace_period", "matched_text": "termination grace period"},
    ]


def test_detects_worker_shutdown_hook_context():
    result = detect_query_graceful_shutdown_requirement("Worker should shutdown gracefully using pre-stop hooks.")

    assert result["requirements"] == [
        {"category": "graceful_shutdown", "matched_text": "shutdown gracefully"},
        {"category": "shutdown_hook", "matched_text": "pre-stop hooks"},
    ]


def test_ignores_unrelated_shutdown_or_power_off_language():
    assert detect_query_graceful_shutdown_requirement(
        "The power-off button should shut down the laptop immediately."
    ) == {"has_graceful_shutdown_requirement": False, "requirements": []}
