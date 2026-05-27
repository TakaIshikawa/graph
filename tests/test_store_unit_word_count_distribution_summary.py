from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_word_count_distribution_summary import summarize_unit_word_count_distribution


def test_unit_word_count_distribution_classifies_bucket_boundaries():
    summary = summarize_unit_word_count_distribution(
        [
            {"source": "notes", "content": "   "},
            {"source": "notes", "content": "word " * 100},
            {"source": "notes", "content": "word " * 101},
            {"source": "notes", "content": "word " * 500},
            {"source": "notes", "content": "word " * 501},
        ]
    )

    assert summary == {
        "total_units": 5,
        "rows": [
            {
                "source": "notes",
                "unit_count": 5,
                "empty_count": 1,
                "short_count": 1,
                "medium_count": 2,
                "long_count": 1,
                "min_words": 0,
                "max_words": 501,
                "average_words": 240.4,
            }
        ],
    }


def test_unit_word_count_distribution_supports_mapping_and_object_text_fields():
    summary = summarize_unit_word_count_distribution(
        [
            {"source": "web", "text": "one two"},
            SimpleNamespace(source="web", body="one two three"),
            {"metadata": {"source": "archive", "body": "one"}},
        ]
    )

    assert summary["rows"] == [
        {
            "source": "archive",
            "unit_count": 1,
            "empty_count": 0,
            "short_count": 1,
            "medium_count": 0,
            "long_count": 0,
            "min_words": 1,
            "max_words": 1,
            "average_words": 1.0,
        },
        {
            "source": "web",
            "unit_count": 2,
            "empty_count": 0,
            "short_count": 2,
            "medium_count": 0,
            "long_count": 0,
            "min_words": 2,
            "max_words": 3,
            "average_words": 2.5,
        },
    ]


def test_unit_word_count_distribution_treats_non_string_content_as_empty():
    summary = summarize_unit_word_count_distribution([{"source": "notes", "content": ["not", "text"]}, {"body": None}])

    assert summary["total_units"] == 2
    assert summary["rows"] == [
        {
            "source": "notes",
            "unit_count": 1,
            "empty_count": 1,
            "short_count": 0,
            "medium_count": 0,
            "long_count": 0,
            "min_words": 0,
            "max_words": 0,
            "average_words": 0.0,
        },
        {
            "source": "unknown",
            "unit_count": 1,
            "empty_count": 1,
            "short_count": 0,
            "medium_count": 0,
            "long_count": 0,
            "min_words": 0,
            "max_words": 0,
            "average_words": 0.0,
        },
    ]


def test_unit_word_count_distribution_sorts_sources_deterministically():
    summary = summarize_unit_word_count_distribution(
        [{"source": "zeta", "content": "one"}, {"source": "Alpha", "content": "one"}, {"source": "alpha", "content": "one"}]
    )

    assert [row["source"] for row in summary["rows"]] == ["Alpha", "alpha", "zeta"]
