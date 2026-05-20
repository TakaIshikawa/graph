from __future__ import annotations

from graph.rag.query_actionability import classify_query_actionability


def test_actionability_blank_query_returns_neutral_structure():
    payload = classify_query_actionability("  \n ")

    assert payload == {
        "normalized_query": "",
        "checklist": False,
        "stepwise": False,
        "recommendation": False,
        "implementation": False,
        "troubleshooting": False,
        "primary_action_type": "none",
        "actionability_level": "none",
        "reasons": {
            "checklist": [],
            "stepwise": [],
            "recommendation": [],
            "implementation": [],
            "troubleshooting": [],
        },
    }


def test_actionability_detects_checklist_and_stepwise_plan():
    payload = classify_query_actionability("Give me a checklist and step-by-step plan")

    assert payload["checklist"] is True
    assert payload["stepwise"] is True
    assert payload["primary_action_type"] == "stepwise"
    assert payload["actionability_level"] == "medium"
    assert payload["reasons"]["checklist"] == ["checklist"]
    assert payload["reasons"]["stepwise"] == ["step by step", "plan"]


def test_actionability_detects_recommendation_and_implementation():
    payload = classify_query_actionability("Which option should we choose and how do we implement it?")

    assert payload["recommendation"] is True
    assert payload["implementation"] is True
    assert payload["primary_action_type"] == "implementation"
    assert payload["actionability_level"] == "high"
    assert payload["reasons"]["recommendation"] == ["should i", "choose"]
    assert payload["reasons"]["implementation"] == ["implement"]


def test_actionability_detects_troubleshooting_as_highest_priority():
    payload = classify_query_actionability("Debug this failing deploy and write steps to fix it")

    assert payload["troubleshooting"] is True
    assert payload["implementation"] is True
    assert payload["stepwise"] is True
    assert payload["primary_action_type"] == "troubleshooting"
    assert payload["actionability_level"] == "high"
    assert payload["reasons"]["troubleshooting"] == ["debug", "fix", "error"]
