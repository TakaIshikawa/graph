from __future__ import annotations

from graph.rag.answer_next_step_owner_coverage import audit_answer_next_step_owner_coverage


def test_audit_answer_next_step_owner_coverage_recognizes_step_formats():
    answer = "\n".join(
        [
            "1. Alice: draft the rollout note.",
            "- Send the evidence packet.",
            "Review metrics assigned to Priya.",
        ]
    )

    result = audit_answer_next_step_owner_coverage(answer)

    assert result["step_count"] == 3
    assert result["owned_step_count"] == 2
    assert result["missing_owner_indexes"] == [1]
    assert result["coverage_score"] == 0.667


def test_audit_answer_next_step_owner_coverage_counts_owner_cues():
    result = audit_answer_next_step_owner_coverage(
        "\n".join(
            [
                "Next owner=research: verify citations.",
                "Action: assigned to Maya for the synthesis.",
                "Update the launch plan with the product team.",
            ]
        )
    )

    assert [cue["type"] for cue in result["owner_cues"]] == ["owner_field", "assigned_to", "team"]
    assert result["owned_step_count"] == 3


def test_audit_answer_next_step_owner_coverage_gives_perfect_score_without_steps():
    assert audit_answer_next_step_owner_coverage("The evidence is descriptive and contains no action plan.") == {
        "step_count": 0,
        "owned_step_count": 0,
        "missing_owner_indexes": [],
        "owner_cues": [],
        "coverage_score": 1.0,
    }
