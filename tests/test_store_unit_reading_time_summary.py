from __future__ import annotations

import pytest

from graph.store.unit_reading_time_summary import summarize_unit_reading_time


def test_reading_time_summary_estimates_minutes_and_buckets_markdown_words():
    report = summarize_unit_reading_time(
        [
            {"content": "# Title\nRead [the guide](https://example.com) now."},
            {"content": " ".join(["word"] * 250)},
            {"content": ""},
        ],
        words_per_minute=100,
    )

    assert report["total_units"] == 3
    assert report["total_words"] == 255
    assert report["zero_word_units"] == 1
    assert report["min_minutes"] == 0
    assert report["max_minutes"] == 3
    assert report["average_minutes"] == 0.85
    assert report["bucket_distribution"] == [
        {"bucket": "0 min", "count": 1},
        {"bucket": "<1 min", "count": 1},
        {"bucket": "1-2 min", "count": 0},
        {"bucket": "3-5 min", "count": 1},
        {"bucket": "6+ min", "count": 0},
    ]


def test_reading_time_summary_requires_positive_wpm():
    with pytest.raises(ValueError):
        summarize_unit_reading_time([], words_per_minute=0)
