from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_units_to_reading_time_estimate_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_reading_time_estimate_csv_counts_words_and_sorts_by_unit_id():
    text = export_units_to_reading_time_estimate_csv(
        [
            {"id": "b", "title": "Beta", "content": "one two\nthree", "source_project": "kindle", "source_entity_type": "highlight"},
            {"id": "a", "title": "Alpha", "content": "", "source_project": "max", "source_entity_type": "note"},
            {"id": "c", "title": "Gamma", "content": "one two three four five", "source_project": "web", "source_entity_type": "article"},
        ],
        words_per_minute=3,
    )

    assert text.splitlines()[0] == "unit_id,title,word_count,estimated_minutes,source,entity_type"
    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "word_count": "0", "estimated_minutes": "0", "source": "max", "entity_type": "note"},
        {"unit_id": "b", "title": "Beta", "word_count": "3", "estimated_minutes": "1", "source": "kindle", "entity_type": "highlight"},
        {"unit_id": "c", "title": "Gamma", "word_count": "5", "estimated_minutes": "2", "source": "web", "entity_type": "article"},
    ]


def test_reading_time_estimate_csv_treats_missing_content_as_zero_words():
    assert rows(export_units_to_reading_time_estimate_csv([{"id": "a", "title": "Untitled"}]))[0]["word_count"] == "0"


def test_reading_time_estimate_csv_path_mode(tmp_path):
    path = tmp_path / "reading-time.csv"
    stats = export_units_to_reading_time_estimate_csv([{"id": "a", "content": "one two"}], path, words_per_minute=2)

    assert path.read_text(encoding="utf-8") == export_units_to_reading_time_estimate_csv([{"id": "a", "content": "one two"}], words_per_minute=2)
    assert stats["rows_exported"] == 1
    assert stats["words_per_minute"] == 2


@pytest.mark.parametrize("words_per_minute", [0, -1, 1.2, True, "200"])
def test_reading_time_estimate_csv_validates_words_per_minute(words_per_minute):
    with pytest.raises(ValueError, match="words_per_minute must be a positive integer"):
        export_units_to_reading_time_estimate_csv([], words_per_minute=words_per_minute)
