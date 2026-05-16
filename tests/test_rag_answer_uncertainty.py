from __future__ import annotations

from graph.rag.answer_uncertainty import annotate_answer_uncertainty


def test_answer_uncertainty_detects_hedges_and_uncited_claims():
    result = annotate_answer_uncertainty(
        "The rollout may finish soon. Revenue reached 42 percent in 2025."
    )

    assert result["counts"] == {
        "sentence_count": 2,
        "annotation_count": 2,
        "result_count": 0,
    }
    assert result["annotations"][0]["cues"] == [
        "hedging_language",
        "uncited_factual_sentence",
    ]
    assert "unsupported_numeric_or_date_claim" in result["annotations"][1]["cues"]
    assert 0 < result["uncertainty_score"] <= 1


def test_answer_uncertainty_keeps_confident_cited_answer_low():
    result = annotate_answer_uncertainty(
        "The rollout finished in 2025 [1].",
        [{"id": "r1", "content": "The rollout finished in 2025."}],
    )

    assert result["annotations"] == []
    assert result["uncertainty_score"] == 0.0


def test_answer_uncertainty_results_improve_support_detection():
    result = annotate_answer_uncertainty(
        "The alpha launch reached pilot customers.",
        [{"id": "r1", "content": "Alpha launch reached pilot customers in April."}],
    )

    assert result["annotations"] == []


def test_answer_uncertainty_detects_conflicting_modality():
    result = annotate_answer_uncertainty("The project must possibly launch tomorrow.")

    assert result["annotations"][0]["cues"] == [
        "hedging_language",
        "uncited_factual_sentence",
        "conflicting_modality",
    ]
    assert result["annotations"][0]["severity"] == "high"
