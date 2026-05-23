from __future__ import annotations

from graph.rag.result_missing_viewpoints import analyze_result_missing_viewpoints


def test_result_missing_viewpoints_scores_complete_set():
    payload = analyze_result_missing_viewpoints(
        "policy rollout",
        [{"text": "government view"}, {"text": "industry response"}, {"text": "public concerns"}, {"text": "expert analysis"}],
    )

    assert payload["missing_viewpoints"] == []
    assert payload["balance_score"] == 1.0


def test_result_missing_viewpoints_flags_partial_set():
    payload = analyze_result_missing_viewpoints("product pricing", [{"text": "customer feedback and analyst note"}])

    assert payload["present_viewpoints"] == ["customer", "analyst"]
    assert payload["missing_viewpoints"] == ["vendor", "competitor"]
    assert len(payload["retrieval_suggestions"]) == 2


def test_result_missing_viewpoints_supports_unspecified_viewpoint_defaults():
    payload = analyze_result_missing_viewpoints("market shift", [{"text": "expert view"}])

    assert payload["present_viewpoints"] == ["expert"]
    assert payload["missing_viewpoints"] == ["supporting", "critical"]
