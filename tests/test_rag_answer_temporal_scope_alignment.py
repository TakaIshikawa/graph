from graph.rag.answer_temporal_scope_alignment import audit_answer_temporal_scope_alignment


def test_missing_temporal_scope_when_evidence_has_dates_but_answer_is_timeless():
    summary = audit_answer_temporal_scope_alignment("The policy applies.", evidence=[{"id": "a", "text": "Updated in 2025."}])

    assert summary["evidence_temporal_marker_count"] == 1
    assert summary["answer_temporal_marker_count"] == 0
    assert summary["missing_temporal_scope"] is True
    assert summary["samples"] == [{"result_id": "a", "matched_temporal_marker": "2025"}]


def test_explicit_answer_date_satisfies_temporal_scope():
    assert audit_answer_temporal_scope_alignment("As of 2025, the policy applies.", [{"text": "Updated in 2025."}])["missing_temporal_scope"] is False


def test_freshness_caveat_satisfies_temporal_scope():
    assert audit_answer_temporal_scope_alignment("Currently, the policy applies.", [{"text": "Updated in 2024."}])["answer_has_freshness_caveat"] is True
