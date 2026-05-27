import pytest

from graph.store import summarize_unit_reading_time_buckets


def test_unit_reading_time_buckets_counts_words_and_samples_stably():
    report = summarize_unit_reading_time_buckets(
        [
            {"id": "zero", "content": "   "},
            {"id": "one", "content": " ".join(["word"] * 100)},
            {"id": "two", "content": " ".join(["word"] * 101)},
            {"id": "long", "content": " ".join(["word"] * 301)},
        ],
        words_per_minute=100,
        minute_buckets=(0, 1, 2),
    )

    assert report["unit_count"] == 4
    assert report["total_words"] == 502
    assert report["average_words_per_unit"] == 125.5
    assert report["buckets"] == [
        {"bucket": "0 min", "count": 1, "samples": [{"unit_id": "zero", "word_count": 0, "minutes": 0}]},
        {
            "bucket": "1-2 min",
            "count": 2,
            "samples": [
                {"unit_id": "one", "word_count": 100, "minutes": 1},
                {"unit_id": "two", "word_count": 101, "minutes": 2},
            ],
        },
        {"bucket": "3+ min", "count": 1, "samples": [{"unit_id": "long", "word_count": 301, "minutes": 4}]},
    ]


def test_unit_reading_time_buckets_validates_configuration():
    with pytest.raises(ValueError):
        summarize_unit_reading_time_buckets([], words_per_minute=0)

    with pytest.raises(ValueError):
        summarize_unit_reading_time_buckets([], minute_buckets=(1, 2))
