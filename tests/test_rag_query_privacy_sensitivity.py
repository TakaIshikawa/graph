from __future__ import annotations

from graph.rag.query_privacy_sensitivity import classify_query_privacy_sensitivity


def test_privacy_sensitivity_blank_query_returns_neutral_structure():
    payload = classify_query_privacy_sensitivity("  \n ")

    assert payload == {
        "normalized_query": "",
        "identity": False,
        "secret": False,
        "financial": False,
        "health": False,
        "location": False,
        "private_communication": False,
        "reasons": {
            "identity": [],
            "secret": [],
            "financial": [],
            "health": [],
            "location": [],
            "private_communication": [],
        },
        "sensitivity_level": "none",
    }


def test_privacy_sensitivity_detects_identity_secret_and_private_communication():
    payload = classify_query_privacy_sensitivity(
        "Find emails from jane@example.com that mention API keys and Slack messages"
    )

    assert payload["identity"] is True
    assert payload["secret"] is True
    assert payload["private_communication"] is True
    assert payload["sensitivity_level"] == "high"
    assert payload["reasons"]["identity"] == ["email address"]
    assert "api key" in payload["reasons"]["secret"]
    assert payload["reasons"]["private_communication"] == ["email", "chat"]


def test_privacy_sensitivity_detects_financial_health_and_location_cues():
    payload = classify_query_privacy_sensitivity(
        "Summarize bank account records, prescriptions, and current location"
    )

    assert payload["financial"] is True
    assert payload["health"] is True
    assert payload["location"] is True
    assert payload["sensitivity_level"] == "high"
    assert payload["reasons"]["financial"] == ["bank account"]
    assert payload["reasons"]["health"] == ["prescription"]
    assert payload["reasons"]["location"] == ["current location"]


def test_privacy_sensitivity_level_is_stable_for_lower_risk_cues():
    identity = classify_query_privacy_sensitivity("lookup passport renewal notes")
    location = classify_query_privacy_sensitivity("notes about home address")
    public = classify_query_privacy_sensitivity("public changelog for retrieval ranking")

    assert identity["sensitivity_level"] == "low"
    assert location["sensitivity_level"] == "low"
    assert public["sensitivity_level"] == "none"
    assert identity["normalized_query"] == "lookup passport renewal notes"
