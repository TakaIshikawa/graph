from __future__ import annotations

from graph.rag.context_window_waste import analyze_context_window_waste


def test_context_window_waste_prefers_explicit_token_counts():
    report = analyze_context_window_waste(
        [{"id": "a", "text": "long text ignored because metadata wins", "metadata": {"token_count": 10}, "score": 0.8}],
        max_tokens=25,
    )

    assert report["used_tokens"] == 10
    assert report["unused_tokens"] == 15
    assert report["over_budget_tokens"] == 0
    assert report["utilization_ratio"] == 0.4


def test_context_window_waste_reports_over_budget_tokens():
    report = analyze_context_window_waste([{"id": "a", "token_count": 9}, {"id": "b", "token_count": 8}], max_tokens=12)

    assert report["used_tokens"] == 17
    assert report["unused_tokens"] == 0
    assert report["over_budget_tokens"] == 5
    assert "trim_context_to_fit_window" in report["recommendations"]


def test_context_window_waste_flags_zero_score_and_small_chunks():
    report = analyze_context_window_waste(
        [{"id": "tiny", "text": "ok", "score": 0}, {"id": "useful", "text": "This chunk has enough text to be useful.", "score": 0.7}],
        max_tokens=100,
    )

    assert report["low_value_chunks"] == [
        {"result_id": "tiny", "token_count": 1, "score": 0.0, "reasons": ["zero_score", "very_small_text_contribution"], "is_low_value": True}
    ]
