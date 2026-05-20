from __future__ import annotations

from dataclasses import dataclass

from graph.rag.evidence_quote_fidelity import score_evidence_quote_fidelity


@dataclass
class EvidenceObject:
    id: str
    quote: str | None = None
    content: str | None = None


def test_evidence_quote_fidelity_exact_and_normalized_matches():
    rows = score_evidence_quote_fidelity(
        [
            {
                "id": "exact",
                "quote": "Revenue increased by 12%.",
                "content": "The filing says Revenue increased by 12%. More detail follows.",
            },
            {
                "id": "normalized",
                "quote": "revenue increased by 12%",
                "content": "Revenue   Increased By 12% in the quarter.",
            },
        ]
    )

    assert rows[0] == {
        "result_id": "exact",
        "fidelity": "exact",
        "score": 1.0,
        "quote": "Revenue increased by 12%.",
        "reason": "quote is an exact substring of source text",
    }
    assert rows[1]["fidelity"] == "normalized"
    assert rows[1]["score"] == 0.85


def test_evidence_quote_fidelity_partial_and_missing_cases():
    rows = score_evidence_quote_fidelity(
        [
            {"id": "partial", "quote": "roadmap alpha rollout", "content": "The roadmap beta rollout moved."},
            {"id": "no-quote", "content": "Only source text exists."},
            {"id": "no-source", "quote": "Only a quote exists."},
        ]
    )

    assert rows[0]["fidelity"] == "partial"
    assert 0.0 < rows[0]["score"] < 0.85
    assert rows[1]["fidelity"] == "missing"
    assert rows[1]["reason"] == "missing quote"
    assert rows[2]["fidelity"] == "missing"
    assert rows[2]["reason"] == "missing source text"


def test_evidence_quote_fidelity_supports_objects_and_tuple_wrapped_payloads():
    rows = score_evidence_quote_fidelity(
        [
            EvidenceObject("object-id", "The launch completed.", "Status: The launch completed."),
            (
                {
                    "result_id": "tuple-id",
                    "snippet": "Service is available",
                    "text": "Latest notice: Service is available for all regions.",
                },
                0.42,
            ),
        ]
    )

    assert rows[0]["result_id"] == "object-id"
    assert rows[0]["fidelity"] == "exact"
    assert rows[1]["result_id"] == "tuple-id"
    assert rows[1]["quote"] == "Service is available"
    assert rows[1]["fidelity"] == "exact"


def test_evidence_quote_fidelity_non_iterable_input_is_empty():
    assert score_evidence_quote_fidelity(None) == []
