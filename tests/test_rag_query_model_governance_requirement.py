from __future__ import annotations

from graph.rag import detect_query_model_governance_requirement


def test_model_governance_detects_core_categories():
    result = detect_query_model_governance_requirement(
        "AI model governance requires model cards, approval workflow, bias testing, human review, "
        "drift monitoring, training data lineage, evaluation metrics, and a rollback plan."
    )

    assert result["has_model_governance_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "approval_workflow",
        "bias_testing",
        "evaluation_metrics",
        "human_review",
        "model_card",
        "monitoring_drift",
        "rollback_plan",
        "training_data_lineage",
    ]


def test_model_governance_requires_ai_or_model_context():
    result = detect_query_model_governance_requirement("Business governance requires approval workflow and rollback plan.")

    assert result["has_model_governance_requirement"] is False
    assert result["requirements"] == []
