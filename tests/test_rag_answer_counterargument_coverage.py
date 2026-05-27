from __future__ import annotations

from graph.rag.answer_counterargument_coverage import audit_answer_counterargument_coverage


def test_detects_limitations_and_alternatives():
    result = audit_answer_counterargument_coverage("We recommend rollout. However, risks remain and alternatives exist.")

    assert result["has_counterarguments"] is True
    assert "limitation" in result["signals"]
    assert "alternative" in result["signals"]


def test_marks_recommendation_missing_counterarguments():
    result = audit_answer_counterargument_coverage("You should adopt the new pipeline because it is faster.")

    assert result["missing_when_expected"] is True


def test_short_factual_answers_do_not_require_counterarguments():
    result = audit_answer_counterargument_coverage("The capital is Paris.")

    assert result["missing_when_expected"] is False
    assert result["signals"] == []
