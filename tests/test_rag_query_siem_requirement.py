from __future__ import annotations

import pytest

from graph.rag.query_siem_requirement import detect_query_siem_requirement


@pytest.mark.parametrize(
    ("query", "categories"),
    [
        ("Is SIEM integration required for enterprise customers?", ["siem_integration"]),
        ("Do we support log forwarding to the customer's SOC?", ["log_forwarding"]),
        ("Can admins export security events for monitoring?", ["event_export"]),
        ("Forward security logs via syslog.", ["log_forwarding", "syslog"]),
        ("Does it integrate with Splunk?", ["splunk"]),
        ("Need Microsoft Sentinel support.", ["sentinel"]),
        ("Confirm IBM QRadar connector support.", ["qradar"]),
        ("Stream security events into our detection pipeline.", ["security_event_streaming"]),
        (
            "Require SIEM support with Splunk, Azure Sentinel, QRadar, syslog, and security event streaming.",
            ["siem_integration", "splunk", "sentinel", "qradar", "syslog", "security_event_streaming"],
        ),
    ],
)
def test_detects_siem_security_event_and_vendor_requirements(query: str, categories: list[str]):
    report = detect_query_siem_requirement(query)

    assert report["requires_siem"] is True
    assert report["categories"] == categories
    assert report["confidence"] == "high"
    assert report["recommendations"] == ["retrieve SIEM integration and security event export documentation"]


@pytest.mark.parametrize(
    "query",
    [
        "How do we turn on application debug logging?",
        "Where are the web server error logs stored?",
        "Add request logging around the checkout controller.",
        "Show database query logs for troubleshooting.",
    ],
)
def test_rejects_generic_application_logging_without_security_event_integration_intent(query: str):
    assert detect_query_siem_requirement(query) == {
        "requires_siem": False,
        "categories": [],
        "matches": [],
        "recommendations": [],
        "confidence": "none",
    }


def test_records_matched_text_and_span_for_first_category_match():
    report = detect_query_siem_requirement("Please export security events to SIEM using syslog.")

    assert report["categories"] == ["event_export", "siem_integration", "syslog"]
    assert report["matches"][0]["matched_text"] == "export security events"
    assert report["matches"][0]["span"] == [7, 29]
