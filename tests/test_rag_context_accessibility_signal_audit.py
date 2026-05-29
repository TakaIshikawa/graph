from graph.rag.context_accessibility_signal import analyze_context_accessibility_signals


def test_detects_positive_accessibility_markers():
    report = analyze_context_accessibility_signals(
        [{"id": "a", "content": "Includes aria-label, captions, transcript, headings, and table headers."}]
    )
    assert report["items"][0]["signals"] == ["alt_text", "captions", "transcript", "headings", "table_headers"]
    assert report["missing_accessibility_count"] == 0


def test_flags_media_only_without_accessible_text():
    report = analyze_context_accessibility_signals(
        [
            {"id": "img", "format": "image", "content": "image-only chart"},
            {"id": "aud", "format": "audio", "content": "audio-only briefing with transcript"},
        ]
    )
    assert report["missing_accessibility_count"] == 1
    assert report["items"][0]["missing_accessibility"] is True
    assert report["items"][1]["missing_accessibility"] is False
