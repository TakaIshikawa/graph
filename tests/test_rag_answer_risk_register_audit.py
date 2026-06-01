from graph.rag.answer_risk_register_audit import audit_answer_risk_register


def test_markdown_table_headers_are_recognized_and_rows_counted():
    result = audit_answer_risk_register(
        """
| Risk | Likelihood | Impact | Mitigation | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| API outage | Medium | High | Add retries | SRE | Open |
| Data drift | Low | Medium | Monitor | ML | Watching |
"""
    )

    assert result == {
        "has_risk_register": True,
        "present_fields": ["risk", "likelihood", "impact", "mitigation", "owner", "status"],
        "missing_fields": [],
        "risk_item_count": 2,
    }


def test_bullet_labels_are_recognized_and_counted():
    result = audit_answer_risk_register(
        """
- Risk: launch delay; Likelihood: medium; Impact: high; Mitigation: reduce scope; Owner: PM; Status: open
- Risk: adoption; Likelihood: low; Impact: medium; Mitigation: training; Owner: CX; Status: tracked
"""
    )

    assert result["has_risk_register"] is True
    assert result["present_fields"] == ["risk", "likelihood", "impact", "mitigation", "owner", "status"]
    assert result["risk_item_count"] == 2


def test_missing_fields_are_reported_in_required_order():
    result = audit_answer_risk_register("- Risk: migration slips; Owner: Platform")

    assert result["has_risk_register"] is True
    assert result["present_fields"] == ["risk", "owner"]
    assert result["missing_fields"] == ["likelihood", "impact", "mitigation", "status"]
    assert result["risk_item_count"] == 1
