from __future__ import annotations

from graph.rag.answer_citation_sequence import audit_answer_citation_sequence


def test_answer_citation_sequence_marks_sequential_citations():
    report = audit_answer_citation_sequence("Alpha is supported [1]. Beta follows [2].")

    assert report == {
        "citation_sequence": [1, 2],
        "first_seen_order": [1, 2],
        "out_of_order_citations": [],
        "repeated_citations": [],
        "citation_count": 2,
        "is_sequential": True,
    }


def test_answer_citation_sequence_reports_out_of_order_first_appearances():
    report = audit_answer_citation_sequence("Beta appears first [2]. Alpha appears later [1].")

    assert report["citation_sequence"] == [2, 1]
    assert report["first_seen_order"] == [2, 1]
    assert report["out_of_order_citations"] == [2, 1]
    assert report["is_sequential"] is False


def test_answer_citation_sequence_handles_grouped_and_repeated_citations():
    report = audit_answer_citation_sequence("Alpha has two sources [1, 3]. Later source two appears [2]. Alpha repeats [1].")

    assert report["citation_sequence"] == [1, 3, 2, 1]
    assert report["first_seen_order"] == [1, 3, 2]
    assert report["out_of_order_citations"] == [3, 2]
    assert report["repeated_citations"] == [1]
    assert report["citation_count"] == 4
    assert report["is_sequential"] is False
