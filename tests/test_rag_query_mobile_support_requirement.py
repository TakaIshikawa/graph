from __future__ import annotations

import pytest

from graph.rag.query_mobile_support_requirement import detect_query_mobile_support_requirement


def test_detects_mobile_platform_cues_and_versions():
    result = detect_query_mobile_support_requirement(
        "Need iOS 17, Android 14, iPadOS, Play Store, and App Store support for the mobile app."
    )

    assert result["requires_mobile_support"] is True
    assert result["cue_categories"] == ["ios", "android", "tablet_support", "app_store_availability", "mobile_app"]
    assert result["platform_values"] == ["iOS 17", "Android 14", "iPadOS", "Play Store", "App Store"]


def test_detects_responsive_ui_mobile_browser_and_minimum_os():
    result = detect_query_mobile_support_requirement(
        "Does it have responsive mobile UI, mobile browser support, and minimum OS version documentation?"
    )

    assert result["requires_mobile_support"] is True
    assert result["cue_categories"] == ["responsive_mobile_ui", "mobile_browser", "minimum_os_version"]


def test_desktop_only_query_does_not_match():
    result = detect_query_mobile_support_requirement("Need desktop compatibility for Windows and macOS.")

    assert result["requires_mobile_support"] is False
    assert result["cue_categories"] == []


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_mobile_support_requirement("   ")
