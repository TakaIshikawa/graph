from __future__ import annotations

from types import SimpleNamespace

from graph.rag.answer_citation_quote_alignment import audit_answer_citation_quote_alignment


def test_answer_citation_quote_alignment_counts_matched_quotes():
    rows = audit_answer_citation_quote_alignment(
        'The report says "substantial improvement" in outcomes.',
        [SimpleNamespace(text="The trial found Substantial Improvement in outcomes.")],
    )

    assert rows == [
        {
            "quote": "substantial improvement",
            "evidence_match_count": 1,
            "severity": "none",
            "claim_text": 'The report says "substantial improvement" in outcomes.',
        }
    ]


def test_answer_citation_quote_alignment_marks_unmatched_quotes_medium():
    rows = audit_answer_citation_quote_alignment(
        'The answer claims "not in evidence".',
        [{"snippet": "A different statement appears here."}],
    )

    assert rows[0]["quote"] == "not in evidence"
    assert rows[0]["evidence_match_count"] == 0
    assert rows[0]["severity"] == "medium"


def test_answer_citation_quote_alignment_suppresses_duplicate_quotes():
    rows = audit_answer_citation_quote_alignment(
        'First "same quote". Later "same quote".',
        [{"content": "same quote"}],
    )

    assert [row["quote"] for row in rows] == ["same quote"]


def test_answer_citation_quote_alignment_reads_dict_evidence_text_fields():
    rows = audit_answer_citation_quote_alignment(
        "One source says 'portable finding'.",
        [{"text": "portable finding"}, {"snippet": "PORTABLE FINDING"}, {"content": "no match"}],
    )

    assert rows[0]["evidence_match_count"] == 2
