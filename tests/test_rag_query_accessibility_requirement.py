from __future__ import annotations

from graph.rag.query_accessibility_requirement import detect_query_accessibility_requirement


def test_recognizes_named_accessibility_standards():
    report = detect_query_accessibility_requirement("Need WCAG 2.2 and ADA evidence for the checkout flow.")

    assert report["requires_accessibility_evidence"] is True
    assert report["standards"] == ["WCAG", "ADA"]
    assert report["accessibility_cues"] == []
    assert report["affected_modalities"] == []
    assert report["recommendations"] == ["retrieve_named_accessibility_standards_and_compliance_evidence"]
    assert report["confidence"] == 0.85


def test_groups_accessibility_cues_by_modality():
    report = detect_query_accessibility_requirement(
        "Check screen reader support, keyboard navigation, captions, alt text, contrast, "
        "colorblind mode, reduced motion, and transcript availability."
    )

    assert [cue["type"] for cue in report["accessibility_cues"]] == [
        "screen_reader",
        "keyboard_navigation",
        "captions",
        "alt_text",
        "contrast",
        "colorblind",
        "reduced_motion",
        "transcript",
    ]
    assert report["affected_modalities"] == ["visual", "audio", "motor", "cognitive"]
    assert report["recommendations"] == [
        "prefer_sources_with_accessibility_testing_or_audit_details",
        "include_visual_accessibility_evidence_such_as_contrast_alt_text_or_screen_reader_support",
        "include_audio_accessibility_evidence_such_as_captions_or_transcripts",
        "include_keyboard_and_focus_navigation_evidence",
        "include_motion_readability_or_cognitive_accessibility_evidence",
    ]
    assert report["confidence"] == 0.8


def test_standards_and_cues_have_high_confidence():
    report = detect_query_accessibility_requirement("Find Section 508 keyboard support and ARIA notes.")

    assert report["standards"] == ["Section 508", "ARIA"]
    assert [cue["type"] for cue in report["accessibility_cues"]] == ["keyboard_navigation"]
    assert report["affected_modalities"] == ["motor"]
    assert report["confidence"] == 0.95


def test_no_cue_query_is_neutral():
    report = detect_query_accessibility_requirement("Summarize the release notes for the latest dashboard.")

    assert report == {
        "requires_accessibility_evidence": False,
        "standards": [],
        "accessibility_cues": [],
        "affected_modalities": [],
        "recommendations": [],
        "confidence": 0.0,
        "normalized_query": "summarize the release notes for the latest dashboard.",
    }
