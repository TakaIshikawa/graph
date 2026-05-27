from graph.rag.result_format_mismatch_audit import audit_result_format_mismatch


def test_result_format_mismatch_audit_detects_requested_and_satisfied_formats():
    result = audit_result_format_mismatch(
        "Return a table and checklist with a concise summary.",
        "\n".join(["| Item | Status |", "| --- | --- |", "| A | Done |", "- [ ] Follow up."]),
    )

    assert result["requested_formats"] == ["table", "checklist", "concise_summary"]
    assert result["satisfied_formats"] == ["table", "checklist", "concise_summary"]
    assert result["missing_formats"] == []
    assert result["mismatch_count"] == 0


def test_result_format_mismatch_audit_reports_missing_formats():
    result = audit_result_format_mismatch("Please provide JSON, CSV, bullets, and a timeline.", "- One item\n- Two items")

    assert result["requested_formats"] == ["json", "csv", "bullets", "timeline"]
    assert result["satisfied_formats"] == ["bullets"]
    assert result["missing_formats"] == ["json", "csv", "timeline"]
    assert result["mismatch_count"] == 3
    assert result["warnings"] == ["missing_json", "missing_csv", "missing_timeline"]


def test_result_format_mismatch_audit_has_no_false_warning_without_format_request():
    assert audit_result_format_mismatch("What changed?", "A short answer.") == {
        "requested_formats": [],
        "satisfied_formats": [],
        "missing_formats": [],
        "mismatch_count": 0,
        "warnings": [],
    }
