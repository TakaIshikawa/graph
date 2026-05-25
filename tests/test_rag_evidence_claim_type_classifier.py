from __future__ import annotations

from graph.rag.evidence_claim_types import classify_evidence_claim_types


def test_evidence_claim_type_classifier_covers_numeric_comparative_causal_and_definitional():
    result = classify_evidence_claim_types(
        [
            {"id": "n", "text": "Revenue increased 12% compared with 2023."},
            {"id": "c", "text": "Delays occurred because the vendor changed the API."},
            {"id": "d", "text": "Latency means the time between request and response."},
        ]
    )

    assert result[0]["claim_types"] == ["numeric", "comparative"]
    assert result[1]["claim_types"] == ["causal", "factual"]
    assert result[2]["claim_types"] == ["definitional"]


def test_evidence_claim_type_classifier_returns_low_confidence_for_ambiguous_snippets():
    result = classify_evidence_claim_types([{"id": "x", "text": "Maybe soon."}])

    assert result[0]["claim_types"] == []
    assert result[0]["confidence"] == "low"
