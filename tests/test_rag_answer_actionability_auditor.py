from __future__ import annotations

from graph.rag import audit_answer_actionability


def test_actionability_auditor_scores_checklist_steps_deadlines_and_prerequisites():
    result = audit_answer_actionability(
        "- Create the rollout ticket by 2025-06-01\n- Assign owner: Platform Team\nRequires staging access.",
        query_intent="action plan",
    )

    assert result["actionability_score"] == 1.0
    assert result["next_steps"] == ["Create the rollout ticket by 2025-06-01", "Assign owner: Platform Team"]
    assert result["deadlines"] == ["2025-06-01"]
    assert result["prerequisites"] == ["staging access"]


def test_actionability_auditor_flags_vague_actions():
    result = audit_answer_actionability("Improve this. Fix things.")

    assert result["vague_action_flags"] == ["Fix things", "Improve this"]
    assert result["actionability_score"] == 0.0
