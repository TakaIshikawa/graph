from __future__ import annotations

from graph.adapters.stackoverflow_favorites_csv import StackOverflowFavoritesCsvAdapter


def test_stackoverflow_favorites_csv_parses_question_metadata_and_tags(tmp_path):
    export = tmp_path / "favorites.csv"
    export.write_text(
        "Question ID,Question Title,URL,Tags,Score,Answer Count,Accepted,Saved At\n"
        '123,How to parse CSV?,https://stackoverflow.com/q/123,"<python><csv>",42,5,true,2025-01-01T00:00:00Z\n',
        encoding="utf-8",
    )

    unit = StackOverflowFavoritesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "question"
    assert unit.metadata["question_id"] == "123"
    assert unit.metadata["url"] == "https://stackoverflow.com/q/123"
    assert unit.metadata["tags"] == ["python", "csv"]
    assert unit.metadata["score"] == 42
    assert unit.metadata["answer_count"] == 5
    assert unit.metadata["accepted"] is True


def test_stackoverflow_favorites_csv_normalizes_tag_formats_and_blank_numbers(tmp_path):
    export = tmp_path / "favorites.csv"
    export.write_text(
        "Title,Tags,Score,Answer Count\n"
        '"One","python; pandas",,\n'
        '"Two","[python, csv]",1,0\n',
        encoding="utf-8",
    )

    units = StackOverflowFavoritesCsvAdapter(path=str(export)).ingest().units

    assert units[0].metadata["tags"] == ["python", "pandas"]
    assert "score" not in units[0].metadata
    assert units[1].metadata["tags"] == ["python", "csv"]
