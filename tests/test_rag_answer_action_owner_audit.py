from graph.rag.answer_action_owner_audit import audit_answer_action_owners


def test_answer_action_owner_audit_counts_markdown_actions_and_owner_cues():
    result = audit_answer_action_owners(
        "\n".join(
            [
                "- [ ] Alice: draft the rollout note.",
                "1. Send the evidence packet.",
                "- Review metrics assigned to Priya.",
                "- Update the launch plan with the product team.",
            ]
        )
    )

    assert result == {
        "action_count": 4,
        "actions_missing_owner": 1,
        "owner_coverage_ratio": 0.75,
        "sampled_actions": ["1. Send the evidence packet."],
        "warnings": ["actions_missing_owner"],
    }


def test_answer_action_owner_audit_is_zero_safe_and_samples_deterministically():
    assert audit_answer_action_owners("This answer has no action plan.") == {
        "action_count": 0,
        "actions_missing_owner": 0,
        "owner_coverage_ratio": 1.0,
        "sampled_actions": [],
        "warnings": [],
    }

    result = audit_answer_action_owners("\n".join(["- Send packet.", "- Create tracker.", "- Review notes.", "- Confirm scope."]))

    assert result["sampled_actions"] == ["- Send packet.", "- Create tracker.", "- Review notes."]
