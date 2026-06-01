from graph.rag.query_sla_credit_requirement import detect_query_sla_credit_requirement


def test_uptime_credit_clauses_are_detected():
    report = detect_query_sla_credit_requirement("Include service credits, uptime credits, and a credit schedule.")

    assert report["requires_sla_credit_terms"] is True
    assert report["credit_terms"] == ["service_credit", "uptime_credit", "credit_schedule"]
    assert report["confidence"] == "high"


def test_refund_penalty_and_breach_compensation_remedies_are_deterministic():
    report = detect_query_sla_credit_requirement("Define SLA penalties, refund for downtime, remedies, and breach of SLA compensation.")

    assert report["credit_terms"] == ["sla_penalty", "refund_for_downtime", "breach_compensation"]
    assert report["remedy_terms"] == ["refund", "penalty", "remedy", "compensation"]


def test_generic_sla_support_hours_do_not_trigger_credit_terms():
    assert detect_query_sla_credit_requirement("What support hours are included in the SLA?") == {
        "requires_sla_credit_terms": False,
        "credit_terms": [],
        "matched_phrases": [],
        "remedy_terms": [],
        "recommendations": [],
        "confidence": "none",
    }
