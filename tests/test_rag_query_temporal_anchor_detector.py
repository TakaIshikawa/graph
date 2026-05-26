from __future__ import annotations

from graph.rag import detect_query_temporal_anchors


def test_temporal_anchor_detector_extracts_years_quarters_and_ranges():
    rows = detect_query_temporal_anchors("Compare 2020 to 2022 and Q2 2024")

    assert rows == [
        {"type": "year_range", "text": "2020 to 2022", "start": "2020-01-01", "end": "2022-12-31"},
        {"type": "quarter", "text": "Q2 2024", "start": "2024-04-01", "end": "2024-06-30"},
    ]


def test_temporal_anchor_detector_resolves_relative_phrases():
    rows = detect_query_temporal_anchors("Plans from last year and next quarter", reference_date="2025-05-15")

    assert rows == [
        {"type": "relative", "text": "last year", "start": "2024-01-01", "end": "2024-12-31"},
        {"type": "relative", "text": "next quarter", "start": "2025-07-01", "end": "2025-09-30"},
    ]


def test_temporal_anchor_detector_returns_empty_list_without_anchors():
    assert detect_query_temporal_anchors("Compare pricing and reliability") == []
