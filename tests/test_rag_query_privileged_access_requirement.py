from __future__ import annotations

from graph.rag.query_privileged_access_requirement import detect_query_privileged_access_requirements


def test_detects_privileged_access_categories():
    rows = detect_query_privileged_access_requirements(
        "Require PAM, just-in-time access, break-glass accounts, administrator approval, "
        "sudo access, and privilege elevation."
    )

    assert [row["category"] for row in rows] == [
        "admin_approval",
        "admin_role",
        "break_glass",
        "jit_access",
        "pam",
        "privilege_elevation",
    ]


def test_deduplicates_repeated_category_matches():
    rows = detect_query_privileged_access_requirements("PAM and privileged access management with JIT access.")

    assert [row["category"] for row in rows] == ["jit_access", "pam"]


def test_avoids_unrelated_admin_examples():
    assert detect_query_privileged_access_requirements("Show admin as a label in the UI examples.") == []
