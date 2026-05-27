from __future__ import annotations

import pytest

from graph.rag.query_platform_requirement import detect_query_platform_requirement


def test_detect_query_platform_requirement_normalizes_explicit_platforms():
    result = detect_query_platform_requirement("Compare iPhone, Android, macOS, Windows, and Linux support.")

    assert result["requires_platform_specificity"] is True
    assert result["platforms"] == ["ios", "android", "macos", "windows", "linux"]
    assert [term["platform"] for term in result["platform_terms"]] == ["ios", "android", "macos", "windows", "linux"]
    assert result["ambiguity_flags"] == []


def test_detect_query_platform_requirement_flags_broad_mobile_and_desktop():
    result = detect_query_platform_requirement("Need the mobile and desktop UX differences.")

    assert result["platforms"] == ["mobile", "desktop"]
    assert result["ambiguity_flags"] == ["broad_mobile", "broad_desktop"]
    assert result["confidence"] == 0.55


def test_detect_query_platform_requirement_detects_api_cli_cloud_and_self_hosted():
    result = detect_query_platform_requirement("Document API, CLI, cloud, and self-hosted deployment options.")

    assert result["platforms"] == ["api", "cli", "cloud", "self_hosted"]
    assert result["requires_platform_specificity"] is True


def test_detect_query_platform_requirement_detects_web_browser_terms():
    result = detect_query_platform_requirement("Check browser behavior in Chrome and Safari for the web app.")

    assert result["platforms"] == ["web", "browser", "chrome", "safari"]
    assert result["ambiguity_flags"] == ["broad_web", "broad_browser"]


def test_detect_query_platform_requirement_leaves_neutral_queries_unflagged():
    result = detect_query_platform_requirement("Summarize the deployment recommendations.")

    assert result["requires_platform_specificity"] is False
    assert result["platforms"] == []
    assert result["platform_terms"] == []


@pytest.mark.parametrize("query", ["", None])
def test_detect_query_platform_requirement_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_platform_requirement(query)  # type: ignore[arg-type]
