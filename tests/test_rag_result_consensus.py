from __future__ import annotations

from dataclasses import dataclass

from graph.rag.result_consensus import analyze_result_consensus


@dataclass
class ResultObject:
    id: str
    claim: str
    source: str


def test_result_consensus_groups_equivalent_claims_across_sources():
    groups = analyze_result_consensus(
        [
            {"id": "a", "claim": "Feature X is available.", "source": "docs"},
            {"id": "b", "title": "Feature X is available", "metadata": {"source": "blog"}},
            {"id": "c", "claim": "Feature Y is delayed.", "source": "docs"},
        ]
    )

    by_claim = {group["normalized_claim"]: group for group in groups}
    feature_x = by_claim["available feature"]

    assert feature_x["result_ids"] == ["a", "b"]
    assert feature_x["source_count"] == 2
    assert feature_x["evidence_count"] == 2
    assert feature_x["consensus_level"] == "multi-source"
    assert feature_x["sources"] == ["blog", "docs"]
    assert by_claim["delayed feature"]["consensus_level"] == "single-source"


def test_result_consensus_detects_conflicting_stances_in_same_group():
    groups = analyze_result_consensus(
        [
            {"id": "old", "claim": "Service A is available", "source": "status", "stance": "support"},
            {"id": "new", "claim": "Service A is available", "source": "incident", "stance": "refuted"},
        ]
    )

    assert groups == [
        {
            "normalized_claim": "available service",
            "result_ids": ["old", "new"],
            "source_count": 2,
            "evidence_count": 2,
            "consensus_level": "conflicting",
            "sources": ["incident", "status"],
        }
    ]


def test_result_consensus_supports_objects_tuple_payloads_and_metadata_claims():
    groups = analyze_result_consensus(
        [
            ResultObject("object-id", "Roadmap beta launched.", "release-notes"),
            (
                {
                    "result_id": "tuple-id",
                    "metadata": {"claim": "Roadmap beta launched.", "source_id": "changelog"},
                },
                0.9,
            ),
        ]
    )

    assert groups[0]["normalized_claim"] == "beta launched roadmap"
    assert groups[0]["result_ids"] == ["object-id", "tuple-id"]
    assert groups[0]["sources"] == ["changelog", "release-notes"]


def test_result_consensus_empty_or_unusable_input_returns_empty_groups():
    assert analyze_result_consensus(None) == []
    assert analyze_result_consensus([{"id": "x"}]) == []
