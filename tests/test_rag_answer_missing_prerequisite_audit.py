from graph.rag.answer_missing_prerequisite_audit import audit_answer_missing_prerequisites


class Evidence:
    def __init__(self, id, text):
        self.id = id
        self.text = text


def test_flags_missing_prerequisite_phrases_from_evidence_shapes():
    rows = audit_answer_missing_prerequisites(
        "Deploy the feature.",
        [
            "Prerequisite: enable audit logging.",
            {"id": "m", "snippet": "Permission required: admin approval."},
            Evidence("o", "Setup step is create a backup."),
        ],
    )

    assert rows == [
        {"prerequisite_phrase": "admin approval", "source_id": "m", "severity": "medium"},
        {"prerequisite_phrase": "create a backup", "source_id": "o", "severity": "medium"},
        {"prerequisite_phrase": "enable audit logging", "source_id": "result-1", "severity": "medium"},
    ]


def test_returns_no_rows_when_answer_mentions_prerequisite():
    assert audit_answer_missing_prerequisites("First enable audit logging.", ["Prerequisite: enable audit logging."]) == []
