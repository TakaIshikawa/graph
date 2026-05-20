from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_reading_time_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, title: str = "", content: str = "", source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_reading_time_csv_estimates_minutes_and_sorts_units():
    text = export_unit_reading_time_csv(
        [
            unit("b", title="Beta", content="one two three four five", source_project="Source B"),
            unit("a", title="Alpha", content=""),
        ],
        words_per_minute=4,
    )

    assert text.splitlines()[0] == "unit_id,title,source,word_count,estimated_minutes,reading_speed_wpm"
    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "Project",
            "word_count": "0",
            "estimated_minutes": "0",
            "reading_speed_wpm": "4",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "Source B",
            "word_count": "5",
            "estimated_minutes": "2",
            "reading_speed_wpm": "4",
        },
    ]


def test_unit_reading_time_csv_uses_minimum_one_minute_for_non_empty_content():
    assert rows(export_unit_reading_time_csv([unit("a", content="short")]))[0]["estimated_minutes"] == "1"


def test_unit_reading_time_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "reading-time.csv"
    units = [unit("a", content="one two")]

    expected = export_unit_reading_time_csv(units, words_per_minute=100)
    stats = export_unit_reading_time_csv(units, path, words_per_minute=100)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "reading_speed_wpm": 100,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("words_per_minute", [0, -1, 1.5, True, "200"])
def test_unit_reading_time_csv_validates_words_per_minute(words_per_minute):
    with pytest.raises(ValueError, match="words_per_minute must be a positive integer"):
        export_unit_reading_time_csv([], words_per_minute=words_per_minute)
