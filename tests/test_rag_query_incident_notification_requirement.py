from graph.rag.query_incident_notification_requirement import detect_query_incident_notification_requirement


def _categories(query):
    return [row["category"] for row in detect_query_incident_notification_requirement(query)]


def test_detect_query_incident_notification_requirement_timeline_channel_and_contact():
    rows = detect_query_incident_notification_requirement(
        "For a security incident, notify customers within 24 hours by email notification and list customer contacts."
    )

    assert [row["category"] for row in rows] == ["customer_contact", "notification_channel", "notification_timeline"]
    assert {row["severity"] for row in rows} == {"high", "medium"}
    assert all(row["evidence"] for row in rows)


def test_detect_query_incident_notification_requirement_severity_regulatory_and_reports():
    assert _categories(
        "Need data breach regulatory notice, critical severity incident thresholds, status updates, and root-cause report requirements."
    ) == ["breach_regulatory_notice", "root_cause_report", "severity_threshold", "status_updates"]


def test_detect_query_incident_notification_requirement_ignores_generic_alerting():
    assert detect_query_incident_notification_requirement("Send monitoring alerts to Slack when CPU usage is high.") == []
