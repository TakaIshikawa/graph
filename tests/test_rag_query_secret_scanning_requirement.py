from graph.rag.query_secret_scanning_requirement import detect_query_secret_scanning_requirements


def test_detects_secret_scanning_categories():
    result = detect_query_secret_scanning_requirements(
        "Need repository scanning and pre-commit scanning with push protection, alerts, remediation, and exceptions."
    )

    assert result["has_secret_scanning_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "alerting",
        "exceptions",
        "prevention",
        "remediation",
        "scan_surface",
    ]


def test_detects_detection_context_cues():
    result = detect_query_secret_scanning_requirements(
        "Compare secret scanning for credential scanning, token leak detection, and exposed API keys."
    )

    assert result["has_secret_scanning_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["detection"]
    assert result["requirements"][0]["matched_text"] == "secret scanning"


def test_requires_secret_scanning_context():
    assert detect_query_secret_scanning_requirements("How should we store application secrets and API keys?") == {
        "has_secret_scanning_requirements": False,
        "requirements": [],
    }
