from graph.rag.answer_evidence_gap_disclosure import audit_answer_evidence_gap_disclosure


def test_missing_disclosure_reports_evidence_gaps():
    result = audit_answer_evidence_gap_disclosure(
        "The answer is complete.",
        [{"id": "a", "snippet": "The source says records are missing."}],
    )

    assert result["evidence_gap_count"] == 1
    assert result["answer_discloses_gap"] is False
    assert result["missing_gap_disclosure"] is True
    assert result["cue_counts"]["missing"] == 1
    assert result["samples"] == [{"result_id": "a", "cue": "missing"}]


def test_disclosed_gap_clears_missing_flag():
    result = audit_answer_evidence_gap_disclosure(
        "The evidence is unclear and some records were not found.",
        [{"id": "a", "content": "Insufficient evidence for the claim."}],
    )

    assert result["evidence_gap_count"] == 1
    assert result["answer_discloses_gap"] is True
    assert result["missing_gap_disclosure"] is False
    assert result["cue_counts"]["insufficient evidence"] == 1


def test_sample_limit_is_clamped_and_limits_samples():
    evidence = [
        {"id": "a", "text": "Unknown result."},
        {"id": "b", "text": "No evidence found."},
    ]

    assert audit_answer_evidence_gap_disclosure("", evidence, sample_limit=1)["samples"] == [
        {"result_id": "a", "cue": "unknown"}
    ]
    assert audit_answer_evidence_gap_disclosure("", evidence, sample_limit=-2)["samples"] == []
