from __future__ import annotations

from graph.rag.answer_unsupported_comparison_audit import audit_answer_unsupported_comparisons


class Evidence:
    def __init__(self, text):
        self.text = text


def test_answer_unsupported_comparisons_marks_grounded_comparison():
    rows = audit_answer_unsupported_comparisons("A is better than B.", [{"snippet": "A is better than B."}])

    assert rows == [{"comparison_sentence": "A is better than B.", "cue": "better than", "evidence_match_count": 1, "severity": None}]


def test_answer_unsupported_comparisons_marks_unsupported_and_deduplicates():
    rows = audit_answer_unsupported_comparisons("A outperforms B. A outperforms B.", [])

    assert rows == [{"comparison_sentence": "A outperforms B.", "cue": "outperforms", "evidence_match_count": 0, "severity": "medium"}]


def test_answer_unsupported_comparisons_supports_string_mapping_and_object_evidence():
    rows = audit_answer_unsupported_comparisons(
        "A has more than B. C versus D is close.",
        ["more than appears here", {"content": "c versus d is close"}, Evidence("unused")],
    )

    assert [row["evidence_match_count"] for row in rows] == [1, 1]
