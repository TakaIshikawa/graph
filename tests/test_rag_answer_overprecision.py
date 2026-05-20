from __future__ import annotations

from graph.rag.answer_overprecision import detect_answer_overprecision


def test_flags_unsupported_precise_numeric_date_currency_and_ranking_claims():
    report = detect_answer_overprecision(
        "Cost was $123.45 on 2025-05-01, rose 12.34%, and ranked #1 with score 9.876.",
        evidence=[{"snippet": "Cost was about $120 and ranked high."}],
        max_decimal_places=2,
    )

    assert report["unsupported_count"] == 5
    assert report["precision_type_counts"]["currency"] == 1
    assert report["precision_type_counts"]["percentage"] == 1
    assert report["precision_type_counts"]["date"] == 1
    assert report["precision_type_counts"]["ranking"] == 1
    assert report["precision_type_counts"]["number"] == 1
    assert all(claim["support_status"] == "unsupported" for claim in report["claims"])


def test_verbatim_evidence_claims_are_not_flagged():
    report = detect_answer_overprecision("Revenue rose 12.34% on 2025-05-01.", evidence=[{"text": "Revenue rose 12.34% on 2025-05-01."}])

    assert report["unsupported_count"] == 0
    assert report["claims"] == []
