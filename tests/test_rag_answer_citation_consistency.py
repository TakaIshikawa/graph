from graph.rag.answer_citation_consistency import audit_answer_citation_consistency


def test_answer_citation_consistency_matches_ids_and_labels():
    report = audit_answer_citation_consistency(
        "Use the cheaper plan [1]. Validate the rollout [A]. Avoid the old flow [missing].",
        [{"id": "ev1", "citation_label": "1"}, {"id": "A"}, "B"],
    )

    assert report == {
        "cited_labels": ["1", "A", "missing"],
        "missing_labels": ["missing"],
        "unused_evidence_ids": ["ev1", "B"],
        "consistency_ratio": 0.6667,
        "findings": [
            {"type": "missing_citation_label", "label": "missing"},
            {"type": "unused_evidence", "evidence_id": "ev1"},
            {"type": "unused_evidence", "evidence_id": "B"},
        ],
    }


def test_answer_citation_consistency_is_zero_safe():
    assert audit_answer_citation_consistency("", []) == {
        "cited_labels": [],
        "missing_labels": [],
        "unused_evidence_ids": [],
        "consistency_ratio": 0.0,
        "findings": [],
    }

    assert audit_answer_citation_consistency("Claim [A].", [])["missing_labels"] == ["A"]
