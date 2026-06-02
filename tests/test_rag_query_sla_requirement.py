from __future__ import annotations

import pytest

from graph.rag.query_sla_requirement import detect_query_sla_requirement


def test_detects_sla_categories_and_targets():
    result = detect_query_sla_requirement(
        "Find SLA terms with 99.9% uptime guarantee, service credits, P1 response in 4 hours, and Sev 1."
    )

    assert result == {
        "requires_sla": True,
        "cue_categories": ["sla", "uptime_guarantee", "service_credits", "severity_levels"],
        "target_values": ["99.9%", "P1", "4 hours", "Sev 1"],
    }


def test_detects_availability_response_and_maintenance_exclusions():
    result = detect_query_sla_requirement(
        "Does the availability target include support response targets by next business day and maintenance exclusions?"
    )

    assert result["requires_sla"] is True
    assert result["cue_categories"] == ["availability_target", "support_response_target", "maintenance_exclusions"]
    assert result["target_values"] == ["next business day"]


def test_generic_reliability_query_without_sla_target_returns_false():
    assert detect_query_sla_requirement("How reliable was the platform last month?") == {
        "requires_sla": False,
        "cue_categories": [],
        "target_values": [],
    }


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_sla_requirement(" ")
