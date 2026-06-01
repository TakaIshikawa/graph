from graph.rag.query_token_revocation_requirement import detect_query_token_revocation_requirements


def test_detects_refresh_and_access_token_revocation():
    result = detect_query_token_revocation_requirements("How do we revoke refresh tokens and invalidate access tokens?")

    assert result["has_token_revocation_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["access_token", "refresh_token"]


def test_detects_logout_denylist_introspection_and_endpoint():
    result = detect_query_token_revocation_requirements(
        "OAuth token revocation endpoint must support logout invalidation, denylist checks, and introspection."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "denylist_blocklist",
        "introspection",
        "logout_invalidation",
        "revocation_endpoint",
    ]


def test_avoids_unrelated_api_token_mentions():
    assert detect_query_token_revocation_requirements("Where do I find the API token for the integration?") == {
        "has_token_revocation_requirements": False,
        "requirements": [],
    }
