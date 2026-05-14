from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_content_length_outliers_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, title: str = "Title", content: str = "content") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_content_length_outliers_csv_empty_input_returns_header():
    assert export_unit_content_length_outliers_csv([]) == (
        "unit_id,title,source_project,source_entity_type,title_length,content_char_count,word_count,bucket\n"
    )


def test_export_unit_content_length_outliers_csv_treats_whitespace_as_empty():
    text = export_unit_content_length_outliers_csv([unit("a", content=" \n\t ")])

    assert rows(text)[0]["bucket"] == "empty"
    assert rows(text)[0]["content_char_count"] == "0"


def test_export_unit_content_length_outliers_csv_excludes_normal_by_default():
    units = [
        unit("empty", content=""),
        unit("short", content=" ".join(["word"] * 30)),
        unit("normal", content=" ".join(["word"] * 100)),
        unit("long", content=" ".join(["word"] * 2001)),
    ]

    assert [row["bucket"] for row in rows(export_unit_content_length_outliers_csv(units))] == [
        "empty",
        "long",
        "short",
    ]


def test_export_unit_content_length_outliers_csv_can_include_normal_rows():
    text = export_unit_content_length_outliers_csv(
        [unit("normal", content=" ".join(["word"] * 100))],
        include_normal=True,
    )

    assert rows(text)[0]["bucket"] == "normal"


def test_export_unit_content_length_outliers_csv_path_mode(tmp_path):
    units = [unit("a", content="")]
    path = tmp_path / "outliers.csv"

    stats = export_unit_content_length_outliers_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["bucket"] == "empty"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
