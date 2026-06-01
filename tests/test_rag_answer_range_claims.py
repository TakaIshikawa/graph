from __future__ import annotations

from graph.rag.answer_range_claims import audit_answer_range_claims


def test_answer_range_claims_extracts_and_matches_ranges():
    rows = audit_answer_range_claims("The lift was 10-20%.", [{"text": "Observed range was 10 to 20%."}])

    assert rows == [{"claim_text": "The lift was 10-20%.", "normalized_range": "10 to 20%", "evidence_match_count": 1, "severity": "none"}]


def test_answer_range_claims_marks_unmatched_ranges():
    rows = audit_answer_range_claims("It ran between 5 and 7 days.", [{"content": "It ran 3 days."}])

    assert rows[0]["normalized_range"] == "5 to 7"
    assert rows[0]["evidence_match_count"] == 0
    assert rows[0]["severity"] == "medium"


def test_answer_range_claims_suppresses_duplicate_sentence_ranges():
    rows = audit_answer_range_claims("From 2019 to 2021, revenue rose from 2019 to 2021.", [{"snippet": "2019 through 2021"}])

    assert [row["normalized_range"] for row in rows] == ["2019 to 2021"]
