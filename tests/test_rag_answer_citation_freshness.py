from __future__ import annotations

from graph.rag import audit_answer_citation_freshness


def test_citation_freshness_computes_age_from_fixed_reference_date():
    rows = audit_answer_citation_freshness(
        "Answer [a] [b]",
        [{"id": "a", "source_date": "2025-01-01"}, {"id": "b", "source_date": "2024-01-01"}],
        reference_date="2025-07-01",
    )

    assert rows == [
        {"citation_id": "b", "source_date": "2024-01-01", "age_days": 547, "severity": "high", "reason": "citation_source_is_stale"},
        {"citation_id": "a", "source_date": "2025-01-01", "age_days": 181, "severity": "medium", "reason": "citation_source_is_aging"},
    ]


def test_citation_freshness_reports_missing_or_invalid_dates():
    rows = audit_answer_citation_freshness("Answer", [{"id": "a"}, {"id": "b", "date": "bad"}], reference_date="2025-01-01")

    assert [row["severity"] for row in rows] == ["high", "high"]
    assert {row["reason"] for row in rows} == {"missing_or_invalid_source_date"}


def test_citation_freshness_sorts_by_severity_then_citation_id():
    rows = audit_answer_citation_freshness(
        "Answer",
        [{"id": "z", "date": "2025-01-01"}, {"id": "a", "date": "2020-01-01"}],
        reference_date="2025-01-02",
    )

    assert [row["citation_id"] for row in rows] == ["a", "z"]
