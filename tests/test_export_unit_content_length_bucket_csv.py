from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_content_length_bucket_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_content_length_bucket_csv_exports_per_unit_boundaries_and_word_counts():
    # Bucket thresholds: empty == 0 chars, short <= 280, medium <= 2,000,
    # long <= 10,000, and very_long > 10,000.
    text = export_units_to_content_length_bucket_csv(
        [
            {"id": "empty", "title": "Empty"},
            {"id": "short", "title": "Short", "content": "one two"},
            {"id": "medium", "title": "Medium", "content": "x" * 281},
            {"id": "long", "title": "Long", "content": "x" * 2001},
            {"id": "very-long", "title": "Very Long", "content": "x" * 10001},
        ]
    )

    assert text.splitlines()[0] == "unit_id,title,char_count,word_count,length_bucket"
    assert rows(text) == [
        {"unit_id": "empty", "title": "Empty", "char_count": "0", "word_count": "0", "length_bucket": "empty"},
        {"unit_id": "long", "title": "Long", "char_count": "2001", "word_count": "1", "length_bucket": "long"},
        {"unit_id": "medium", "title": "Medium", "char_count": "281", "word_count": "1", "length_bucket": "medium"},
        {"unit_id": "short", "title": "Short", "char_count": "7", "word_count": "2", "length_bucket": "short"},
        {
            "unit_id": "very-long",
            "title": "Very Long",
            "char_count": "10001",
            "word_count": "1",
            "length_bucket": "very_long",
        },
    ]


def test_content_length_bucket_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "unit-content-length-buckets.csv"
    units = [{"id": "a", "title": "A", "content": "alpha beta"}]

    expected = export_units_to_content_length_bucket_csv(units)
    stats = export_units_to_content_length_bucket_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
