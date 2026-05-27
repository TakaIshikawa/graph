from __future__ import annotations

from graph.rag.answer_example_coverage_audit import audit_answer_example_coverage


class Evidence:
    def __init__(self, text):
        self.text = text


def test_answer_example_coverage_marks_supported_examples():
    rows = audit_answer_example_coverage("Use open data, for example census tables.", [{"snippet": "Census tables support this example."}])

    assert rows == [{"example_sentence": "Use open data, for example census tables.", "cue": "for example", "evidence_match_count": 1, "severity": None}]


def test_answer_example_coverage_marks_unsupported_and_deduplicates():
    rows = audit_answer_example_coverage("It includes tools including payroll exports. It includes tools including payroll exports.", [])

    assert rows == [{"example_sentence": "It includes tools including payroll exports.", "cue": "including", "evidence_match_count": 0, "severity": "medium"}]


def test_answer_example_coverage_supports_string_mapping_and_object_evidence():
    rows = audit_answer_example_coverage(
        "Sources such as incident reports are useful. E.g. audit logs help too.",
        ["incident reports are useful", {"content": "audit logs help too"}, Evidence("unused")],
    )

    assert [row["evidence_match_count"] for row in rows] == [1, 1]
