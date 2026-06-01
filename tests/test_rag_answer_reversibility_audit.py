from graph.rag.answer_reversibility_audit import audit_answer_reversibility


def test_flags_high_impact_actions_without_reversibility_language():
    rows = audit_answer_reversibility("Delete the old index. Upgrade the cluster. Rotate the keys.")

    assert rows == [
        {"action_text": "Delete the old index", "missing_reversibility_signal": True, "severity": "high"},
        {"action_text": "Rotate the keys", "missing_reversibility_signal": True, "severity": "high"},
        {"action_text": "Upgrade the cluster", "missing_reversibility_signal": True, "severity": "high"},
    ]


def test_suppresses_when_neighboring_sentence_mentions_rollback_or_backup():
    assert audit_answer_reversibility("Create a backup first. Delete the old index. Restore it if needed.") == []
