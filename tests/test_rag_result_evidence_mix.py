from __future__ import annotations

from dataclasses import dataclass

from graph.rag.result_evidence_mix import analyze_result_evidence_mix


@dataclass
class ResultStub:
    id: str
    metadata: dict


def test_analyze_result_evidence_mix_computes_counts_and_percentages():
    payload = analyze_result_evidence_mix(
        [
            {
                "id": "paper",
                "source_type": "paper",
                "confidence": 0.9,
                "relation_type": "supports",
                "published_at": "2026-05-01",
            },
            {
                "id": "news",
                "source_type": "news",
                "confidence": 0.55,
                "relation_type": "mentions",
            },
            {
                "id": "blog",
                "source_type": "blog",
                "confidence": 0.2,
                "relation_type": "supports",
                "metadata": {"updated_at": "2026-04-01T12:00:00Z"},
            },
        ]
    )

    assert payload["total_results"] == 3
    assert payload["counts"] == {
        "source_types": {"blog": 1, "news": 1, "paper": 1},
        "confidence_buckets": {"high": 1, "low": 1, "medium": 1},
        "relation_types": {"mentions": 1, "supports": 2},
        "date_coverage": {"dated": 2, "undated": 1},
    }
    assert payload["percentages"]["date_coverage"] == {"dated": 66.7, "undated": 33.3}
    assert payload["imbalances"] == []


def test_analyze_result_evidence_mix_flags_missing_and_dominant_categories():
    payload = analyze_result_evidence_mix(
        [
            {"id": "a", "source_type": "blog", "relation_type": "supports"},
            {"id": "b", "source_type": "blog", "relation_type": "supports"},
            {"id": "c", "source_type": "blog", "relation_type": "supports"},
            {"id": "d"},
        ]
    )

    assert payload["counts"]["source_types"] == {"blog": 3, "unknown": 1}
    assert payload["counts"]["confidence_buckets"] == {"unknown": 4}
    assert payload["counts"]["date_coverage"] == {"undated": 4}
    assert payload["imbalances"] == [
        "missing confidence buckets metadata",
        "dominant confidence buckets: unknown",
        "dominant date coverage: undated",
        "missing relation types metadata",
        "dominant relation types: supports",
        "missing source types metadata",
        "dominant source types: blog",
        "missing date coverage",
    ]


def test_analyze_result_evidence_mix_handles_empty_inputs_deterministically():
    assert analyze_result_evidence_mix([]) == {
        "total_results": 0,
        "counts": {
            "source_types": {},
            "confidence_buckets": {},
            "relation_types": {},
            "date_coverage": {},
        },
        "percentages": {
            "source_types": {},
            "confidence_buckets": {},
            "relation_types": {},
            "date_coverage": {},
        },
        "imbalances": ["no evidence results"],
    }


def test_analyze_result_evidence_mix_supports_objects_and_nested_metadata():
    payload = analyze_result_evidence_mix(
        [
            ResultStub(
                id="object",
                metadata={
                    "kind": "memo",
                    "score": "80",
                    "relation": "contradicts",
                    "date": "2026-01-01",
                },
            ),
            {
                "unit": {
                    "metadata": {
                        "type": "dataset",
                        "source_confidence": 0.7,
                        "edge_type": "supports",
                    }
                }
            },
        ]
    )

    assert payload["counts"]["source_types"] == {"dataset": 1, "memo": 1}
    assert payload["counts"]["confidence_buckets"] == {"high": 1, "medium": 1}
    assert payload["counts"]["relation_types"] == {"contradicts": 1, "supports": 1}
    assert payload["counts"]["date_coverage"] == {"dated": 1, "undated": 1}
