from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_word_count_distribution_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_word_count_distribution_csv_empty_input_returns_header():
    assert export_unit_word_count_distribution_csv([]) == "bucket,min_words,max_words,unit_count,unit_ids\n"


def test_export_unit_word_count_distribution_csv_buckets_counts_numerically():
    text = export_unit_word_count_distribution_csv(
        [
            {"id": "empty", "content": "   "},
            {"id": "short", "content": "one two three"},
            {"id": "long", "content": "word " * 101},
        ]
    )

    assert rows(text) == [
        {"bucket": "0", "min_words": "0", "max_words": "0", "unit_count": "1", "unit_ids": "empty"},
        {"bucket": "1-100", "min_words": "1", "max_words": "100", "unit_count": "1", "unit_ids": "short"},
        {"bucket": "101-500", "min_words": "101", "max_words": "500", "unit_count": "1", "unit_ids": "long"},
    ]


def test_export_unit_word_count_distribution_csv_path_mode(tmp_path):
    path = tmp_path / "word-counts.csv"
    stats = export_unit_word_count_distribution_csv([{"id": "a", "content": "hello world"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["bucket"] == "1-100"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
