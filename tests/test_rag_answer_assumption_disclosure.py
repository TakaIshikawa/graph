from __future__ import annotations

from graph.rag.answer_assumption_disclosure import audit_answer_assumption_disclosure


def test_identifies_explicit_assumption_section():
    result = audit_answer_assumption_disclosure("Assumptions: usage grows 10%. The estimate is $12k.")

    assert result["assumption_count"] == 1
    assert result["disclosed_assumptions"][0]["cue"] == "Assumptions"
    assert result["needs_disclosure"] is False


def test_identifies_inline_assumption_phrase():
    result = audit_answer_assumption_disclosure("The cloud option is cheaper if we assume storage remains flat.")

    assert result["disclosed_assumptions"] == [{"type": "assumption_phrase", "cue": "if we assume", "span": [28, 40]}]
    assert result["needs_disclosure"] is False


def test_flags_implicit_cues_without_disclosure():
    result = audit_answer_assumption_disclosure("This implies churn rose because onboarding changed.")

    assert result["assumption_count"] == 0
    assert result["implicit_cues"] == [{"type": "this_implies", "cue": "This implies", "span": [0, 12]}]
    assert result["needs_disclosure"] is True


def test_handles_answers_without_assumptions_without_false_positives():
    result = audit_answer_assumption_disclosure("TLS encrypts network traffic.")

    assert result == {
        "assumption_count": 0,
        "disclosed_assumptions": [],
        "implicit_cues": [],
        "needs_disclosure": False,
    }
