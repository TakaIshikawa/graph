from graph.rag.result_accessibility_coverage import analyze_result_accessibility_coverage


def test_accessibility_hits_include_id_index_and_signals():
    report = analyze_result_accessibility_coverage(
        [
            {"id": "a", "title": "WCAG audit", "content": "Covers alt text and keyboard navigation."},
            {"id": "b", "content": "Performance notes."},
        ]
    )
    assert report["total_results"] == 2
    assert report["accessibility_result_count"] == 1
    assert report["accessibility_ratio"] == 0.5
    assert report["accessibility_results"] == [{"id": "a", "index": 0, "signals": ["WCAG", "alt_text", "keyboard_navigation"]}]


def test_accessibility_query_without_coverage_recommends_more_evidence():
    report = analyze_result_accessibility_coverage([{"id": "x", "content": "No relevant notes."}], query="Need WCAG coverage")
    assert report["accessibility_ratio"] == 0.0
    assert report["recommendation"] == "retrieve_sources_with_accessibility_evidence"
