from __future__ import annotations

from graph.rag import audit_answer_numeric_claims


def test_numeric_claim_auditor_detects_integers_decimals_percentages_and_years():
    rows = audit_answer_numeric_claims("Revenue rose 12.5% in 2024. There were 42 teams.", ["2024 filing confirms 12.5%."])

    assert [row["normalized_number"] for row in rows] == ["2024", "12.5%", "42"]
    assert rows[0]["severity"] == "none"
    assert rows[1]["severity"] == "none"
    assert rows[2]["severity"] == "medium"


def test_numeric_claim_auditor_supported_numbers_are_not_flagged():
    assert audit_answer_numeric_claims("Latency is 50 ms.", [{"snippet": "The benchmark reports 50 ms."}])[0]["severity"] == "none"


def test_numeric_claim_auditor_deduplicates_claims_deterministically():
    rows = audit_answer_numeric_claims("Use 10 samples. Use 10 samples.", [])

    assert len(rows) == 1
    assert rows[0]["normalized_number"] == "10"
