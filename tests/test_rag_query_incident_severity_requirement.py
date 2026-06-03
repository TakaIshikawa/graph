from graph.rag.query_incident_severity_requirement import detect_query_incident_severity_requirement


def test_detects_sev1_definitions_and_escalation_thresholds():
    report = detect_query_incident_severity_requirement(
        "Find the incident severity matrix with Sev1 definition, Sev2 criteria, "
        "and escalation thresholds for the on-call team."
    )

    assert report["requires_incident_severity"] is True
    assert report["severity_terms"] == ["severity_level", "sev_definition", "severity_matrix"]
    assert report["escalation_terms"] == ["escalation_threshold"]
    assert report["matched_phrases"] == ["severity matrix", "Sev1 definition", "Sev2 criteria", "escalation thresholds"]
    assert report["confidence"] == "high"
    assert "severity matrix" in report["recommendations"][0]
    assert "response-time or escalation threshold" in report["recommendations"][-1]


def test_detects_priority_impact_tiers_and_response_times():
    report = detect_query_incident_severity_requirement(
        "For outages, compare P1/P2 priority classification, customer impact tiers, "
        "blast radius, and response times by severity."
    )

    assert report == {
        "requires_incident_severity": True,
        "severity_terms": ["priority_classification", "impact_tier"],
        "escalation_terms": ["response_time"],
        "matched_phrases": ["P1", "P2", "priority classification", "customer impact tiers", "blast radius", "response times by severity"],
        "recommendations": [
            "Retrieve a severity matrix mapping incident levels to impact and urgency.",
            "Include evidence for impact tiers such as customer impact, business impact, or blast radius.",
            "Include severity-based response-time or escalation threshold evidence.",
        ],
        "confidence": "high",
    }


def test_general_postmortem_question_is_not_a_severity_requirement():
    assert detect_query_incident_severity_requirement("Summarize the incident postmortem and lessons learned.") == {
        "requires_incident_severity": False,
        "severity_terms": [],
        "escalation_terms": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }


def test_generic_priority_without_incident_context_is_not_enough():
    assert detect_query_incident_severity_requirement("Prioritize the roadmap features for next quarter.") == {
        "requires_incident_severity": False,
        "severity_terms": [],
        "escalation_terms": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }
