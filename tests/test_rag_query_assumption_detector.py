from __future__ import annotations

from graph.rag.query_assumption_detector import detect_query_assumptions


def test_query_assumption_detector_detects_premise_loaded_query():
    payload = detect_query_assumptions("Why did the API continue to fail since the rollout?")

    assert payload["assumptions"] == ["why did", "since", "continue to"]
    assert payload["assumption_count"] == 3
    assert payload["risk_level"] == "high"


def test_query_assumption_detector_handles_neutral_query():
    payload = detect_query_assumptions("What changed in the API rollout?")

    assert payload == {
        "assumptions": [],
        "assumption_count": 0,
        "verification_questions": [],
        "risk_level": "low",
    }


def test_query_assumption_detector_handles_multi_assumption_query():
    payload = detect_query_assumptions("Given that sales are still falling again, what happened?")

    assert payload["assumptions"] == ["given that", "still", "again"]
    assert len(payload["verification_questions"]) == 3
