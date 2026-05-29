from graph.rag.answer_step_order import audit_answer_step_order


def test_passes_well_ordered_numbered_procedure():
    report = audit_answer_step_order("1. Prepare\n2. Run\n3. Verify")
    assert report["ordered"] is True
    assert report["issues"] == []


def test_flags_duplicate_skipped_and_decreasing_numbers():
    report = audit_answer_step_order("1. Start\n1. Repeat\n3. Skip\n2. Back")
    assert [issue["type"] for issue in report["issues"]] == ["duplicate_step_number", "skipped_step_number", "decreasing_step_number"]
    assert report["ordered"] is False


def test_detects_conflicting_sequence_words():
    report = audit_answer_step_order("Finally deploy. First run the tests.")
    assert report["issues"] == [{"type": "conflicting_sequence_marker", "from": "finally", "to": "first"}]
