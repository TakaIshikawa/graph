from __future__ import annotations

from dataclasses import dataclass

from graph.rag.evidence_entity_overlap import analyze_evidence_entity_overlap


@dataclass
class Result:
    id: str
    text: str
    metadata: dict


def test_evidence_entity_overlap_extracts_text_and_metadata_entities():
    report = analyze_evidence_entity_overlap(
        "How did Acme Cloud affect ada@example.com and #Launch?",
        [
            {"id": "a", "content": "Acme Cloud launched #Launch.", "metadata": {"entities": ["Ada"]}},
            {"id": "b", "snippet": "Other Corp contacted ada@example.com."},
        ],
    )

    assert report["rows"][0]["shared_entities"] == ["#Launch", "Acme Cloud"]
    assert report["rows"][0]["missing_query_entities"] == ["ada@example.com"]
    assert report["rows"][0]["overlap_score"] == 0.666667
    assert report["rows"][1]["shared_entities"] == ["ada@example.com"]


def test_evidence_entity_overlap_accepts_objects_tuples_and_claims():
    report = analyze_evidence_entity_overlap(
        "Tell me about Grid",
        [(Result("obj", "Grid Alpha uses Battery Pack.", {"tags": ["Grid"]}), 0.8)],
        claims=[{"claim": "Battery Pack shipped."}],
    )

    assert report["rows"][0]["shared_entities"] == ["Battery Pack", "Grid"]
    assert report["summary"] == {"query_entity_count": 2}
