from graph.rag.query_model_evaluation_requirement import detect_query_model_evaluation_requirement


def test_evaluation_phrases_trigger_model_evaluation():
    result = detect_query_model_evaluation_requirement("Require a golden dataset, regression eval, benchmark suite, and quality gate.")

    assert result["requires_model_evaluation"] is True
    assert result["evaluation_types"] == ["benchmark_suite", "golden_dataset", "quality_gate", "regression_eval"]
    assert result["confidence"] == "high"


def test_metrics_are_captured_when_present():
    result = detect_query_model_evaluation_requirement("Run RAG evals with precision, recall, latency, and pass rate targets.")

    assert result["requires_model_evaluation"] is True
    assert result["evaluation_types"] == ["evals"]
    assert result["metrics"] == ["latency", "pass_rate", "precision", "recall"]


def test_generic_model_question_returns_defaults():
    assert detect_query_model_evaluation_requirement("Which model should answer customer support questions?") == {
        "requires_model_evaluation": False,
        "evaluation_types": [],
        "metrics": [],
        "matched_cues": [],
        "confidence": "none",
    }
