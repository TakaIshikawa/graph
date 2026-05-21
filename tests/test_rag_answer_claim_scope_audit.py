from __future__ import annotations

from types import SimpleNamespace

from graph.rag.answer_claim_scope_audit import audit_answer_claim_scope


def test_flags_broad_claim_supported_only_by_limited_evidence():
    report = audit_answer_claim_scope(
        "The treatment always works for everyone.",
        [{"id": "trial", "text": "A small preliminary pilot found some patients may improve."}],
    )

    assert report["total_claims"] == 1
    assert report["overstated_claims"] == 1
    assert report["reason_counts"] == {"evidence_narrows_scope": 1, "missing_matching_scope_support": 1}
    assert report["claims"][0]["narrowing_result_ids"] == ["trial"]


def test_broad_claim_with_matching_scope_evidence_is_scoped():
    report = audit_answer_claim_scope(
        "The rule is guaranteed for all supported plans.",
        [{"id": "manual", "text": "The rule is guaranteed for all supported plans."}],
    )

    assert report["scoped_claims"] == 1
    assert report["overstated_claims"] == 0
    assert report["reason_counts"] == {}


def test_supports_object_and_tuple_wrapped_results_and_empty_inputs():
    result = (SimpleNamespace(id="obj", text="A limited sample suggests some benefit."), 0.9)
    report = audit_answer_claim_scope("It is proven for all users.", [result])

    assert report["claims"][0]["narrowing_result_ids"] == ["obj"]
    assert "evidence_narrows_scope" in report["claims"][0]["reasons"]

    empty = audit_answer_claim_scope("", [])
    assert empty["warnings"] == ["no_answer", "no_results"]
    assert empty["claims"] == []
