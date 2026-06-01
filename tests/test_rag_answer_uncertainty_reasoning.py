from graph.rag.answer_uncertainty_reasoning import audit_answer_uncertainty_reasoning


def test_explained_uncertainty_has_reason():
    summary = audit_answer_uncertainty_reasoning("It may change because evidence is limited.")

    assert summary["uncertainty_sentence_count"] == 1
    assert summary["unexplained_uncertainty_count"] == 0
    assert summary["samples"][0]["has_reason"] is True


def test_unexplained_uncertainty_is_flagged():
    summary = audit_answer_uncertainty_reasoning("This could fail. The rollout is complete.")

    assert summary["has_unexplained_uncertainty"] is True
    assert summary["samples"][0]["sentence_index"] == 0


def test_answer_without_uncertainty_returns_empty_samples():
    assert audit_answer_uncertainty_reasoning("The system stores three backups.") == {
        "uncertainty_sentence_count": 0,
        "unexplained_uncertainty_count": 0,
        "has_unexplained_uncertainty": False,
        "samples": [],
    }
