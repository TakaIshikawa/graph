from __future__ import annotations

from types import SimpleNamespace

from graph.rag.claim_traceability import map_claim_traceability


def test_claim_traceability_matches_explicit_citations_and_sources():
    trace = map_claim_traceability(
        [
            {"id": "c1", "text": "Alpha shipped.", "citation_ids": ["r1"]},
            {"id": "c2", "text": "Beta was sourced.", "source_ids": ["source-b"]},
        ],
        [
            {"id": "r1", "content": "Other text", "source_id": "source-a"},
            {"id": "r2", "content": "Other text", "source_id": "source-b"},
        ],
    )

    assert trace["claims"][0]["matches"][0] == {
        "match_type": "citation",
        "score": 1.0,
        "result_id": "r1",
        "source_id": "source-a",
    }
    assert trace["claims"][1]["matches"][0]["result_id"] == "r2"
    assert trace["support_counts"]["explicit_match_count"] == 2


def test_claim_traceability_uses_content_overlap_for_uncited_claims():
    trace = map_claim_traceability(
        ["The roadmap includes alpha launch milestones."],
        [{"id": "r1", "content": "Alpha launch milestones are tracked in the roadmap."}],
    )

    assert trace["claims"][0]["matches"] == [
            {
                "match_type": "content_overlap",
                "score": 0.8,
                "result_id": "r1",
                "source_id": None,
            }
    ]
    assert trace["unsupported_claims"] == []


def test_claim_traceability_surfaces_unsupported_claims_and_objects():
    claim = SimpleNamespace(id="obj-claim", text="Gamma was cancelled.")
    result = SimpleNamespace(id="obj-result", text="Alpha launched.", source_id="src")

    trace = map_claim_traceability([claim], [result])

    assert trace["claims"][0]["supported"] is False
    assert trace["unsupported_claims"] == [
        {"claim_id": "obj-claim", "claim": "Gamma was cancelled."}
    ]
    assert trace["support_counts"]["unsupported_count"] == 1


def test_claim_traceability_no_results():
    trace = map_claim_traceability(["A claim"], [])

    assert trace["claims"][0]["matches"] == []
    assert trace["support_counts"] == {
        "claim_count": 1,
        "result_count": 0,
        "supported_count": 0,
        "unsupported_count": 1,
        "explicit_match_count": 0,
        "overlap_match_count": 0,
    }
