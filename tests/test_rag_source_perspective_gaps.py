from __future__ import annotations

from graph.rag.source_perspective_gaps import analyze_source_perspective_gaps


def test_source_perspective_gaps_balanced_inputs_have_no_missing_recommendation_gaps():
    report = analyze_source_perspective_gaps(
        "compare vector databases",
        [
            {"id": "u", "perspective": "user_community"},
            {"id": "a", "source_type": "academic"},
            {"id": "n", "url": "https://news.example/story", "content": "news analysis"},
        ],
    )

    assert report["missing_perspectives"] == []
    assert report["warnings"] == []


def test_source_perspective_gaps_single_vendor_flags_missing_independent_views():
    report = analyze_source_perspective_gaps("best vendor for search", [{"id": "v", "perspective": "vendor"}])

    assert report["counts"] == {"vendor": 1}
    assert report["missing_perspectives"] == ["user_community", "academic", "news_media"]
    assert report["warnings"] == ["missing_recommended_perspectives"]


def test_source_perspective_gaps_unknown_source_is_reported():
    report = analyze_source_perspective_gaps("what is bm25", [{"id": "x", "content": "plain note"}])

    assert report["perspectives"] == [{"result_id": "x", "perspective": "unknown"}]
    assert report["warnings"] == ["unknown_perspectives"]
