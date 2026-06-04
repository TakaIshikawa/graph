from __future__ import annotations

from graph.rag import detect_query_cookie_consent_requirement


def test_cookie_consent_detects_banner_reject_categories_preferences_and_logs():
    result = detect_query_cookie_consent_requirement(
        "For GDPR web privacy, require a cookie banner, reject all, cookie categories, "
        "preference center, consent logs, and third-party cookies."
    )

    assert result["has_cookie_consent_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "consent_banner",
        "consent_logging",
        "cookie_categories",
        "preference_center",
        "reject_all",
        "third_party_cookies",
    ]
    assert all(row["matched_text"] and row["severity"] for row in result["requirements"])


def test_cookie_consent_detects_prior_consent():
    result = detect_query_cookie_consent_requirement("Do we need prior consent before setting cookies?")

    assert [row["category"] for row in result["requirements"]] == ["prior_consent"]


def test_browser_cookie_implementation_without_privacy_context_does_not_trigger():
    result = detect_query_cookie_consent_requirement("How do I set a browser cookie with SameSite in JavaScript?")

    assert result["has_cookie_consent_requirement"] is False
    assert result["requirements"] == []
