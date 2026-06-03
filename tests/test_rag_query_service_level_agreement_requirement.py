from __future__ import annotations

from graph.rag.query_service_level_agreement_requirement import detect_query_service_level_agreement_requirements


def test_detects_saas_vendor_sla_requirements_sorted_by_category():
    result = detect_query_service_level_agreement_requirements(
        "For a SaaS vendor contract, find 99.9% uptime SLA terms, support response time, "
        "service credits, maintenance window, and availability target evidence."
    )

    assert result["has_service_level_agreement_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "availability_target",
        "maintenance_window",
        "service_credits",
        "support_response_time",
        "uptime_sla",
    ]
    assert result["requirements"][0]["matched_text"] == "99.9%"
    assert result["requirements"][4]["matched_text"] == "99.9%"


def test_numeric_availability_target_is_captured_in_matched_text():
    result = detect_query_service_level_agreement_requirements(
        "Does the vendor provide a 99.95% availability commitment for the platform?"
    )

    assert result["requirements"] == [
        {"category": "availability_target", "severity": "high", "matched_text": "99.95%", "span": (26, 45)}
    ]


def test_detects_support_response_and_service_credit_terms():
    result = detect_query_service_level_agreement_requirements(
        "Review the provider SLA for response window remedies and service credits."
    )

    assert [row["category"] for row in result["requirements"]] == ["service_credits", "support_response_time"]


def test_casual_availability_question_without_service_context_does_not_trigger():
    result = detect_query_service_level_agreement_requirements("Are you available tomorrow around 99.9% sure?")

    assert result == {
        "has_service_level_agreement_requirements": False,
        "requirements": [],
    }
