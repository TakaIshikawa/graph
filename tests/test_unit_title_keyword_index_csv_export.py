from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_unit_title_keyword_index_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, title: str, source_project: str = "Project", tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="content",
        metadata={},
        tags=tags or [],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_title_keyword_index_csv_indexes_keywords_and_orders_by_count():
    text = export_unit_title_keyword_index_csv(
        [
            unit("a", title="The SQLite edge guide", source_project="A", tags=["db"]),
            unit("b", title="SQLite on the server", source_project="B", tags=["server"]),
            unit("c", title="AI notes", tags=["ml"]),
        ]
    )

    assert rows(text)[:2] == [
        {
            "keyword": "sqlite",
            "unit_count": "2",
            "sources": "A; B",
            "tags": "db; server",
            "unit_titles": "SQLite on the server; The SQLite edge guide",
        },
        {
            "keyword": "edge",
            "unit_count": "1",
            "sources": "A",
            "tags": "db",
            "unit_titles": "The SQLite edge guide",
        },
    ]
    assert "the" not in {row["keyword"] for row in rows(text)}


def test_unit_title_keyword_index_csv_honors_min_length():
    keywords = [row["keyword"] for row in rows(export_unit_title_keyword_index_csv([unit("a", title="AI ML and SQL")], min_length=2))]

    assert keywords == ["ai", "ml", "sql"]


def test_unit_title_keyword_index_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "keywords.csv"
    units = [unit("a", title="SQLite guide")]

    expected = export_unit_title_keyword_index_csv(units)
    stats = export_unit_title_keyword_index_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 2


@pytest.mark.parametrize("min_length", [0, -1, 1.2, True, "3"])
def test_unit_title_keyword_index_csv_validates_min_length(min_length):
    with pytest.raises(ValueError, match="min_length must be a positive integer"):
        export_unit_title_keyword_index_csv([], min_length=min_length)
