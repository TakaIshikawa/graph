from graph.rag.query_availability_zone_redundancy_requirement import (
    detect_query_availability_zone_redundancy_requirements,
)


def test_detects_multi_az_and_availability_zone_redundancy_queries():
    rows = detect_query_availability_zone_redundancy_requirements("Require multi-AZ availability zone redundancy.")

    assert rows == [
        {
            "matched_text": "availability zone redundancy",
            "category": "availability_zone_redundancy",
            "requirement_strength": "high",
        },
        {"matched_text": "multi-AZ", "category": "multi_az", "requirement_strength": "high"},
    ]


def test_detects_active_active_region_failover_and_zone_failure_tolerance():
    rows = detect_query_availability_zone_redundancy_requirements(
        "Need active-active deployment, regional failover, and zone failure tolerance."
    )

    assert rows == [
        {"matched_text": "active-active deployment", "category": "active_active_deployment", "requirement_strength": "high"},
        {"matched_text": "regional failover", "category": "region_failover", "requirement_strength": "high"},
        {
            "matched_text": "zone failure tolerance",
            "category": "single_az_failure_tolerance",
            "requirement_strength": "high",
        },
    ]


def test_unrelated_scaling_or_cloud_hosting_query_does_not_match():
    assert detect_query_availability_zone_redundancy_requirements("Compare autoscaling for generic cloud hosting.") == []
