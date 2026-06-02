from __future__ import annotations

from graph.rag.query_siem_integration_requirement import detect_query_siem_integration_requirements


def test_detects_named_siem_products_and_export_requirements():
    rows = detect_query_siem_integration_requirements(
        "Need SIEM integration evidence for Splunk, Microsoft Sentinel, and IBM QRadar export."
    )

    assert [row["category"] for row in rows] == ["siem_export", "splunk", "sentinel", "qradar"]


def test_detects_protocol_streaming_and_alert_enrichment_without_duplicate_categories():
    rows = detect_query_siem_integration_requirements(
        "Can it forward logs via syslog, provide a security event stream, and add alert enrichment context for syslog?"
    )

    assert [row["category"] for row in rows] == ["syslog_forwarding", "security_event_streaming", "alert_enrichment"]


def test_generic_logging_without_security_event_intent_does_not_match():
    assert detect_query_siem_integration_requirements("How do we turn on application debug logging?") == []
