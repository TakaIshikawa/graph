from __future__ import annotations

from graph.rag.query_vendor_lock_in_requirement import detect_query_vendor_lock_in_requirements


def test_detect_query_vendor_lock_in_requirements_category_coverage():
    rows = detect_query_vendor_lock_in_requirements(
        "Avoid vendor lock-in, proprietary APIs, define an exit strategy, document the migration path, and estimate switching costs."
    )

    assert [row["category"] for row in rows] == [
        "lock_in",
        "proprietary_dependency",
        "exit_strategy",
        "migration_path",
        "switching_cost",
    ]
    assert {row["severity"] for row in rows} == {"high", "medium"}


def test_detect_query_vendor_lock_in_requirements_sorts_by_first_match():
    rows = detect_query_vendor_lock_in_requirements("Compare cost to switch, vendor exit, and locked in risks.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("switching_cost", "cost to switch"),
        ("exit_strategy", "vendor exit"),
        ("lock_in", "locked in"),
    ]


def test_detect_query_vendor_lock_in_requirements_ignores_unrelated_vendor_mentions():
    assert detect_query_vendor_lock_in_requirements("Compare vendor pricing and support response times.") == []
