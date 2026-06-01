from __future__ import annotations

from graph.rag.query_password_policy_requirement import detect_query_password_policy_requirements


def test_detects_password_policy_categories():
    rows = detect_query_password_policy_requirements(
        "Need password length, password complexity, rotation, password reuse history, "
        "breached password screening, and account lockout."
    )

    assert [row["category"] for row in rows] == [
        "breach_screening",
        "complexity",
        "length",
        "lockout",
        "reuse_history",
        "rotation",
    ]


def test_handles_hyphenated_and_whitespace_varied_phrases():
    rows = detect_query_password_policy_requirements("Set min-14 characters and change passwords every 90 days.")

    assert [row["category"] for row in rows] == ["length", "rotation"]


def test_avoids_unrelated_password_examples():
    assert detect_query_password_policy_requirements("The sample variable is named password in a code example.") == []
