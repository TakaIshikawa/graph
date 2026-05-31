from graph.rag.answer_confidence_calibration_audit import audit_answer_confidence_calibration


def test_uncertainty_scores_higher_than_absolute_claims():
    calibrated = audit_answer_confidence_calibration("The result likely applies, but there is limited evidence.")
    absolute = audit_answer_confidence_calibration("This is definitely the only possible answer and cannot be wrong.")

    assert calibrated["uncertainty_marker_count"] >= 2
    assert absolute["overconfidence_flags"]
    assert calibrated["calibration_score"] > absolute["calibration_score"]


def test_extracts_numeric_confidence_phrases():
    result = audit_answer_confidence_calibration("I have 80% confidence based on the cited study.")

    assert result["confidence_claim_count"] == 1
    assert result["numeric_confidence_percentages"] == [80]


def test_empty_answer_returns_low_score_and_zero_counts():
    result = audit_answer_confidence_calibration("")

    assert result["calibration_score"] == 0.0
    assert result["confidence_claim_count"] == 0
    assert result["uncertainty_marker_count"] == 0
    assert result["overconfidence_flags"] == []
