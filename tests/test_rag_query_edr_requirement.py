from graph.rag.query_edr_requirement import detect_query_edr_requirements


def test_detects_edr_categories():
    result = detect_query_edr_requirements(
        "EDR needs agent coverage, endpoint telemetry, behavioral detection, isolation, threat hunting, and SIEM alerts."
    )

    assert result["has_edr_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "alert_integration",
        "detection",
        "endpoint_coverage",
        "response_actions",
        "telemetry",
        "threat_hunting",
    ]


def test_detects_quarantine_and_endpoint_detection_response_context():
    result = detect_query_edr_requirements("Endpoint detection and response should quarantine hosts.")

    assert result["has_edr_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["detection", "response_actions"]


def test_requires_edr_context():
    assert detect_query_edr_requirements("Summarize endpoint monitoring uptime dashboards.") == {
        "has_edr_requirements": False,
        "requirements": [],
    }
