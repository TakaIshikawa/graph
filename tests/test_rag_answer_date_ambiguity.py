from __future__ import annotations

from graph.rag.answer_date_ambiguity import audit_answer_date_ambiguity


def test_answer_date_ambiguity_flags_relative_dates_without_anchor():
    rows = audit_answer_date_ambiguity("Today the policy applies. It changed recently and may expand soon.")

    assert [row["phrase"] for row in rows] == ["today", "recently", "soon"]
    assert all(row["severity"] == "medium" for row in rows)


def test_answer_date_ambiguity_ignores_relative_dates_with_absolute_anchor():
    rows = audit_answer_date_ambiguity("Today, June 1, 2026, the rule applies. Last year in 2025, it did not.")

    assert rows == []
