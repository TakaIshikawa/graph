from __future__ import annotations

from graph.rag import detect_query_breach_notification_requirement


def test_breach_notification_detects_deadline_regulator_customer_data_and_severity():
    result = detect_query_breach_notification_requirement(
        "For a data breach, notify regulators and affected customers within 72 hours, "
        "include affected data, and define severity thresholds."
    )

    assert result["has_breach_notification_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "affected_data_scope",
        "customer_notice",
        "incident_severity_threshold",
        "notification_deadline",
        "regulator_notice",
    ]


def test_breach_notification_detects_delay_and_template():
    result = detect_query_breach_notification_requirement(
        "Security incident response should cover law enforcement delay and notification templates."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "communication_template",
        "law_enforcement_delay",
    ]


def test_generic_outage_notification_without_breach_context_does_not_trigger():
    result = detect_query_breach_notification_requirement("Send customers an outage notice within 2 hours.")

    assert result["has_breach_notification_requirement"] is False
    assert result["requirements"] == []
