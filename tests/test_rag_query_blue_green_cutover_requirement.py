from __future__ import annotations

from graph.rag.query_blue_green_cutover_requirement import detect_query_blue_green_cutover_requirement


def test_detects_blue_green_cutover_categories():
    result = detect_query_blue_green_cutover_requirement(
        "For the service release, require blue-green cutover, validate the green environment, "
        "traffic switch, DNS cutover, keep the old environment, and instant rollback."
    )

    assert result["has_blue_green_cutover_requirement"] is True
    assert result["requirements"] == [
        {"category": "blue_green_deployment", "matched_text": "blue-green cutover", "severity": "high"},
        {"category": "green_environment", "matched_text": "green environment", "severity": "medium"},
        {"category": "instant_rollback", "matched_text": "instant rollback", "severity": "high"},
        {"category": "old_environment_retention", "matched_text": "keep the old environment", "severity": "medium"},
        {"category": "route_cutover", "matched_text": "DNS cutover", "severity": "high"},
        {"category": "traffic_switch", "matched_text": "traffic switch", "severity": "high"},
    ]


def test_detects_route_and_rollback_cutover_wording():
    result = detect_query_blue_green_cutover_requirement(
        "Production deployment should switch traffic with a load balancer flip and rollback to blue."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "instant_rollback",
        "route_cutover",
        "traffic_switch",
    ]


def test_ignores_unrelated_color_and_environment_language():
    assert detect_query_blue_green_cutover_requirement(
        "Compare blue and green branding themes for an environmental report."
    ) == {"has_blue_green_cutover_requirement": False, "requirements": []}
