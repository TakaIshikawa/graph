from graph.rag.evidence_retraction_signals import audit_evidence_retraction_signals


def test_detects_title_retraction_cue():
    report = audit_evidence_retraction_signals([{"id": "a", "title": "Retraction: original article"}])

    assert report["has_retraction_signals"] is True
    assert report["signal_counts"]["retraction"] == 1
    assert report["flagged_results"][0]["source_id"] == "a"


def test_detects_snippet_withdrawal_cue():
    report = audit_evidence_retraction_signals([{"snippet": "This preprint was withdrawn by the authors."}])

    assert report["signal_counts"]["withdrawal"] == 1
    assert report["flagged_results"][0]["source_id"] == "result-1"


def test_detects_metadata_status_and_notes():
    report = audit_evidence_retraction_signals(
        [{"metadata": {"status": "expression of concern", "notes": "journal notice"}}]
    )

    assert report["signal_counts"]["expression_of_concern"] == 1


def test_classifies_correction_separately_from_retraction():
    report = audit_evidence_retraction_signals([{"title": "Correction to trial results"}])

    assert report["signal_counts"]["correction"] == 1
    assert report["signal_counts"]["retraction"] == 0
