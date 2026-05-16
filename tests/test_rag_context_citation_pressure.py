from __future__ import annotations

from dataclasses import dataclass

from graph.rag.context_citation_pressure import estimate_context_citation_pressure


@dataclass
class Result:
    text: str
    metadata: dict


def test_context_citation_pressure_scores_factual_context():
    report = estimate_context_citation_pressure(
        [
            {"content": "Acme Cloud revenue rose 18% in 2025. Grid Alpha had 42 incidents.", "url": "https://a.test/x"},
            {"snippet": "Battery Pack shipments reached 120 units on 2025-03-01.", "url": "https://b.test/y"},
        ]
    )

    assert report["label"] == "high"
    assert report["recommended_min_citations"] == 2
    assert report["contributing_factors"] == {
        "factual_sentence_count": 3,
        "numeric_count": 7,
        "date_count": 2,
        "entity_count": 3,
        "source_count": 2,
    }


def test_context_citation_pressure_accepts_objects_tuples_and_empty():
    report = estimate_context_citation_pressure([(Result("Short note.", {"source": "docs"}), 0.5)])

    assert report["label"] == "low"
    assert estimate_context_citation_pressure([])["recommended_min_citations"] == 0
