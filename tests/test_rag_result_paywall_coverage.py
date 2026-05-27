from __future__ import annotations

from graph.rag.result_paywall_coverage import analyze_result_paywall_coverage


def test_paywall_coverage_counts_access_statuses():
    report = analyze_result_paywall_coverage(
        [
            {"id": "a", "access": "open access"},
            {"id": "b", "paywall": True},
            {"id": "c", "notes": "login required"},
            {"id": "d"},
        ]
    )

    assert report["counts"] == {"open": 1, "paywalled": 1, "login_required": 1, "unknown": 1}
    assert report["ratios"]["paywalled"] == 0.25
    assert report["paywalled_result_ids"] == ["b"]


def test_paywall_coverage_supports_boolean_and_string_metadata():
    report = analyze_result_paywall_coverage([{"id": "oa", "is_open_access": True}, {"id": "p", "access": "subscription required"}])

    assert report["results"] == [{"id": "oa", "access_status": "open"}, {"id": "p", "access_status": "paywalled"}]
