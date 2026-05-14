from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_content_length_summary_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    content: str | None = "",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content=content,
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_content_length_summary_empty_input_returns_header_only_csv():
    assert export_unit_content_length_summary_csv([]) == (
        "source_project,source_entity_type,unit_count,empty_content_count,min_length,"
        "median_length,max_length,average_length\n"
    )


def test_unit_content_length_summary_groups_and_calculates_odd_median():
    text = export_unit_content_length_summary_csv(
        [
            unit("a", content="one"),
            unit("b", content="  two\nwords "),
            unit("c", content=""),
            unit("d", source_project=SourceProject.PINBOARD, source_entity_type="bookmark", content="link"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "unit_count": "3",
            "empty_content_count": "1",
            "min_length": "0",
            "median_length": "3.00",
            "max_length": "9",
            "average_length": "4.00",
        },
        {
            "source_project": "pinboard",
            "source_entity_type": "bookmark",
            "unit_count": "1",
            "empty_content_count": "0",
            "min_length": "4",
            "median_length": "4.00",
            "max_length": "4",
            "average_length": "4.00",
        },
    ]


def test_unit_content_length_summary_handles_even_median_and_unknown_fallbacks():
    text = export_unit_content_length_summary_csv(
        [
            unit("a", source_project=None, source_entity_type=None, content="aa"),
            unit("b", source_project="", source_entity_type="", content="aaaa"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "unit_count": "2",
            "empty_content_count": "0",
            "min_length": "2",
            "median_length": "3.00",
            "max_length": "4",
            "average_length": "3.00",
        }
    ]


def test_unit_content_length_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "length-summary.csv"
    units = [unit("a", content="content")]

    expected = export_unit_content_length_summary_csv(units)
    stats = export_unit_content_length_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_unit_content_length_summary_is_deterministic_for_reversed_input():
    units = [
        unit("a", source_project="Source B", content="aaa"),
        unit("b", source_project="Source A", content="a"),
        unit("c", source_project="Source A", source_entity_type="task", content="aa"),
    ]

    assert export_unit_content_length_summary_csv(units) == export_unit_content_length_summary_csv(
        reversed(units)
    )
