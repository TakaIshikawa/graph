from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_claim_redundancy import analyze_result_claim_redundancy


def test_groups_exact_and_near_duplicate_claims_deterministically():
    result = analyze_result_claim_redundancy(
        [{"id": "a", "claim": "Latency improved for checkout users"}, {"id": "b", "summary": "Checkout latency improved for users"}]
    )

    assert len(result["redundant_groups"]) == 1
    assert result["redundant_groups"][0]["result_ids"] == ["a", "b"]


def test_distinct_claims_remain_unique():
    result = analyze_result_claim_redundancy([{"claim": "Latency improved"}, {"claim": "Costs increased"}])

    assert result["redundant_groups"] == []
    assert result["unique_claim_count"] == 2


def test_supports_object_style_results():
    result = analyze_result_claim_redundancy([SimpleNamespace(id="x", claims=["Revenue rose quickly"]), {"id": "y", "title": "Revenue rose quickly"}])

    assert result["redundant_groups"][0]["result_ids"] == ["x", "y"]
