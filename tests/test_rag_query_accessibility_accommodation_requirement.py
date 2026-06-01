from __future__ import annotations

from graph.rag.query_accessibility_accommodation_requirement import (
    detect_query_accessibility_accommodation_requirements,
)


def test_detect_query_accessibility_accommodation_requirements_category_coverage():
    rows = detect_query_accessibility_accommodation_requirements(
        "Need WCAG evidence, screen reader support, keyboard-only operation, captions, alt text, color contrast, and reduced motion."
    )

    assert [row["category"] for row in rows] == [
        "wcag",
        "screen_reader",
        "keyboard_only",
        "captions_transcripts",
        "alt_text",
        "contrast",
        "reduced_motion",
    ]


def test_detect_query_accessibility_accommodation_requirements_acronym_matching():
    rows = detect_query_accessibility_accommodation_requirements("Validate wcag with NVDA and JAWS.")

    assert [(row["category"], row["matched_text"]) for row in rows] == [
        ("wcag", "wcag"),
        ("screen_reader", "nvda"),
    ]


def test_detect_query_accessibility_accommodation_requirements_empty_query():
    assert detect_query_accessibility_accommodation_requirements("") == []
