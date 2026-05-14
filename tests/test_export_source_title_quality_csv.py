from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_title_quality_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str = "Title",
    content: str = "Content",
    source_project: object = SourceProject.MAX,
    source_entity_type: str = "note",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title,
        content=content,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_title_quality_csv_empty_input_returns_header():
    assert export_source_title_quality_csv([]) == (
        "source_project,source_entity_type,total_units,blank_title_count,blank_title_rate,"
        "duplicate_title_count,duplicate_title_rate,title_content_duplicate_count,"
        "title_content_duplicate_rate,very_short_title_count,very_short_title_rate\n"
    )


def test_export_source_title_quality_csv_groups_with_unknown_fallbacks():
    text = export_source_title_quality_csv([unit("a", source_project="", source_entity_type="")])

    assert rows(text)[0]["source_project"] == "Unknown"
    assert rows(text)[0]["source_entity_type"] == "Unknown"


def test_export_source_title_quality_csv_counts_quality_signals_by_group():
    text = export_source_title_quality_csv(
        [
            unit("a", title=" Same  Title ", content="Body"),
            unit("b", title="same title", content="Body"),
            unit("c", title="", content="Body"),
            unit("d", title="abc", content="abc"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "total_units": "4",
            "blank_title_count": "1",
            "blank_title_rate": "25.00",
            "duplicate_title_count": "2",
            "duplicate_title_rate": "50.00",
            "title_content_duplicate_count": "1",
            "title_content_duplicate_rate": "25.00",
            "very_short_title_count": "1",
            "very_short_title_rate": "25.00",
        }
    ]


def test_export_source_title_quality_csv_duplicate_counts_are_group_scoped():
    text = export_source_title_quality_csv(
        [
            unit("a", title="Same", source_entity_type="note"),
            unit("b", title="Same", source_entity_type="bookmark"),
        ]
    )

    assert [row["duplicate_title_count"] for row in rows(text)] == ["0", "0"]


def test_export_source_title_quality_csv_path_mode(tmp_path):
    units = [unit("a", title="")]
    path = tmp_path / "titles.csv"

    stats = export_source_title_quality_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["blank_title_count"] == "1"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
