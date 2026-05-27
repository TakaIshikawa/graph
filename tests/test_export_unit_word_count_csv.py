from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_word_count_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_word_count_csv_counts_multiline_markdown():
    rows = _rows(export_units_to_word_count_csv([{"id": "u1", "title": "One", "content": "# Heading\nA short paragraph."}]))

    assert rows == [
        {
            "unit_id": "u1",
            "title": "One",
            "word_count": "4",
            "line_count": "2",
            "paragraph_count": "1",
            "character_count": "28",
        }
    ]


def test_word_count_csv_blank_lines_separate_paragraphs_without_extra_words():
    rows = _rows(export_units_to_word_count_csv([{"id": "u1", "content": "Alpha beta\n\n\nGamma"}]))

    assert rows[0]["word_count"] == "3"
    assert rows[0]["line_count"] == "4"
    assert rows[0]["paragraph_count"] == "2"


def test_word_count_csv_handles_empty_content():
    rows = _rows(export_units_to_word_count_csv([{"id": "u1", "title": "Empty", "content": ""}]))

    assert rows[0]["word_count"] == "0"
    assert rows[0]["line_count"] == "0"
    assert rows[0]["paragraph_count"] == "0"
    assert rows[0]["character_count"] == "0"
