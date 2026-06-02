from __future__ import annotations

from graph.rag.query_network_segmentation_requirement import detect_query_network_segmentation_requirement


def test_detects_segmentation_boundary_categories():
    result = detect_query_network_segmentation_requirement(
        "Show network segmentation, microsegmentation, subnet isolation, east-west traffic controls, private networking, and zero-trust network boundaries."
    )

    assert result == {
        "requires_network_segmentation": True,
        "cue_categories": [
            "network_segmentation",
            "microsegmentation",
            "subnet_isolation",
            "east_west_controls",
            "private_networking",
            "zero_trust_boundaries",
        ],
    }


def test_overlapping_segmentation_phrases_are_deduplicated():
    result = detect_query_network_segmentation_requirement("Network segmentation and segmented network controls.")

    assert result["cue_categories"] == ["network_segmentation"]


def test_general_networking_question_does_not_match():
    assert detect_query_network_segmentation_requirement("Compare network throughput between regions.") == {
        "requires_network_segmentation": False,
        "cue_categories": [],
    }
