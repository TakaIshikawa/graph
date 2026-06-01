from __future__ import annotations

from graph.rag.query_explainability_requirement import detect_query_explainability_requirements


def test_detect_query_explainability_requirements_category_coverage():
    rows = detect_query_explainability_requirements(
        "Need explainability, interpretable outputs, include rationale, decision reasons, model transparency, and recommendation rationale."
    )

    assert [row["category"] for row in rows] == [
        "explainability",
        "interpretability",
        "rationale",
        "decision_reasons",
        "model_transparency",
        "recommendation_rationale",
    ]
    assert all(row["matched_text"] for row in rows)


def test_detect_query_explainability_requirements_transparent_reasoning_and_why():
    rows = detect_query_explainability_requirements(
        "Show transparent reasoning and why a recommendation was made."
    )

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("model_transparency", "transparent reasoning"),
        ("recommendation_rationale", "why a recommendation was made"),
    ]


def test_detect_query_explainability_requirements_ignores_ordinary_explanation():
    assert detect_query_explainability_requirements("Explain how to reset a password.") == []
