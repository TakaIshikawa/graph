from __future__ import annotations

from graph.rag.query_authorization_scope_requirement import detect_query_authorization_scope_requirements


def test_detects_authorization_scope_categories():
    rows = detect_query_authorization_scope_requirements(
        "Compare RBAC, ABAC, least privilege, role permissions, scoped tokens, "
        "admin-only access, row-level security, and permission boundaries."
    )

    assert [row["category"] for row in rows] == [
        "abac",
        "admin_only",
        "least_privilege",
        "permission_boundary",
        "rbac",
        "role_permissions",
        "row_level_security",
        "scoped_tokens",
    ]


def test_deduplicates_overlapping_wording_by_category():
    rows = detect_query_authorization_scope_requirements(
        "RBAC and role-based access control with scoped tokens and token scopes."
    )

    assert [row["category"] for row in rows] == ["rbac", "scoped_tokens"]


def test_whitespace_normalized_mixed_case_query_matches():
    rows = detect_query_authorization_scope_requirements("Need   RoW-Level   Security and ADMIN ONLY policy.")

    assert [row["category"] for row in rows] == ["admin_only", "row_level_security"]


def test_unrelated_query_returns_empty_list():
    assert detect_query_authorization_scope_requirements("Summarize authentication examples.") == []
